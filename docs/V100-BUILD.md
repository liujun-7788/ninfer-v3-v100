**English** | [简体中文](V100-BUILD.zh-CN.md)

# Building and running NInfer on Tesla V100 (sm_70) — verified recipe

This guide reflects a build that is actually running in production on a
Tesla V100-PCIe-32GB (PCIe, not SXM). Getting here took a full day of
fixes; everything below is the distilled, verified path.

Verified environment:

| Component | Version |
|---|---|
| GPU | Tesla V100-PCIe-32GB (sm_70), driver 580 |
| OS | Ubuntu 24.04 |
| CUDA toolkit (nvcc) | 12.8 (12.8.93), installed side-by-side with system CUDA 12.0 |
| CMake | 3.28.3 |
| CUTLASS | 4.4.2 (local tarball) |
| Libraries | libcurl 8.5, ffmpeg 60.x (libavformat/libavcodec/libavutil/libswscale) |

## TL;DR

```bash
# 1. system packages
sudo apt-get install -y build-essential cmake pkg-config \
  libcurl4-openssl-dev libavformat-dev libavcodec-dev \
  libavutil-dev libswscale-dev

# 2. CUDA toolkit 12.8+ (nvcc). Our system toolkit was 12.0; we installed
#    12.8 alongside it (silent toolkit install leaves the driver untouched)
#    and pointed CMake at it explicitly.
#    https://developer.nvidia.com/cuda-downloads

# 3. sources
git clone https://github.com/liujun-7788/ninfer-v3-v100.git
cd ninfer-v3-v100

# 4. CUTLASS 4.4.2 as a local directory (see pitfall 2)
wget https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.4.2.tar.gz \
  -O cutlass-4.4.2.tar.gz
tar xzf cutlass-4.4.2.tar.gz
# if the tarball extracts double-nested (cutlass-4.4.2/cutlass-4.4.2/),
# flatten it so that ./cutlass-4.4.2/CMakeLists.txt exists

# 5. configure + build (sm_70 only — this is the single biggest time saver)
cmake -B build-v100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DFETCHCONTENT_SOURCE_DIR_CUTLASS=$PWD/cutlass-4.4.2
cmake --build build-v100 -j$(nproc)

# binary: build-v100/apps/ninfer-serve
```

## The pitfalls (what cost us a day)

1. **`CMAKE_CUDA_ARCHITECTURES=70`.** A default configure builds for a long
   list of architectures (70 through 121a). Pass `70` explicitly — the V100
   fork's kernels hard-check compute capability 7.0 anyway (an RTX 4090,
   sm_89, cannot run this build and is rejected at startup; do not try to
   substitute other cards), and the build gets dramatically faster.

2. **CUTLASS via FetchContent.** If your network cannot reach GitHub (our
   production box cannot), the CMake FetchContent download hangs or fails.
   Download the v4.4.2 tarball manually and point
   `-DFETCHCONTENT_SOURCE_DIR_CUTLASS` at the extracted directory. Watch
   out: the tag tarball can extract **double-nested**
   (`cutlass-4.4.2/cutlass-4.4.2/`) — flatten it so that
   `<dir>/CMakeLists.txt` exists, otherwise configure fails.

3. **CUDA toolkit version.** We verified with toolkit 12.8 (nvcc 12.8.93)
   while the system default was 12.0. The upstream project targets
   Blackwell (sm_120a) and assumes 12.8; rather than fighting a mixed
   toolchain, install 12.8 side-by-side (it does not touch the driver) and
   pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`.

4. **nvlink `Skipping incompatible ...` warnings** for libdl / librt /
   libpthread during linking are harmless — the engine links and runs
   fine. Do not chase them.

5. **Media frontends.** Image/video input needs ffmpeg dev libraries
   (libavformat/libavcodec/libavutil/libswscale) and libcurl, detected via
   pkg-config at configure time. Install them before the cmake step if you
   want more than text I/O.

## Runtime flags that matter on V100

A production line we run daily (Qwen3.8-27B NVFP4, official v3 artifact):

```bash
build-v100/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 7105 --device 0 --model-id qwen3.8-27b \
  --max-context 131072 --kv-capacity auto --max-concurrency 1 \
  --kv-dtype int8 --device-state-slots 1 --host-state-slots 8 \
  --host-kv-mib 8192 --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking --pending-timeout-ms 600000
```

- **CUDA Graphs work on V100** (capture ~0.3–0.7 s at these settings). If
  capture OOMs, your KV budget is too aggressive for the free VRAM: lower
  `--max-context` first; only fall back to `--no-cuda-graph` as a last
  resort.
- **MTP draft tokens (K).** We swept K=2..5 on Qwen3.8-27B NVFP4 over a
  real HTTP workload (context 2K/8K/32K/64K/128K, 3 repetitions each,
  identical prompts across K):

  | K | avg decode tok/s (5 ctx) | 2K | 8K | 32K | 64K | 128K | median acceptance |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 2 | 78.2 | 80.7 | 119.6 | 85.1 | 70.6 | 35.2 | 71.3% |
  | **3** | **84.2** | **123.0** | **115.4** | 78.7 | 56.4 | **47.5** | 70.0% |
  | 4 | 61.7 | 73.3 | 75.5 | 73.6 | 54.8 | 31.1 | 42.9% |
  | 5 | 77.4 | 109.7 | 90.3 | 85.1 | 69.8 | 32.2 | 66.1% |

  **K=3 is the best overall default.** K=2 wins at 64K context; K=4
  collapses (acceptance drops to ~40–47% and throughput with it). Prefill
  is unaffected by K (measured spread <1% across K at every context size).

- **Prefill is serialized.** The engine runs one prefill at a time;
  concurrent requests queue and receive HTTP 503 after
  `--pending-timeout-ms` (default 30 s) if prefill takes longer. Raise it
  for production (we use 600000), or keep `--max-concurrency 1`.
- **Cold start takes minutes** on first launch (weights paging + graph
  capture). Probe readiness via the HTTP endpoint (`GET /v1/models`),
  not via systemd state — the unit can look "active" while the engine is
  still not listening.

## Models

Official upstream v3 artifacts load directly with this fork, e.g.
[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).
v2 artifacts (magic `NInfer\0\2`) keep working unchanged.
