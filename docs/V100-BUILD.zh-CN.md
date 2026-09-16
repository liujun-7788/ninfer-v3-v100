[English](V100-BUILD.md) | **简体中文**

# 在 Tesla V100 (sm_70) 上编译并运行 NInfer——经验证的完整指南

本指南反映的是一套在 Tesla V100-PCIe-32GB（PCIe 版，非 SXM）上**真实生产运行**的构建。走到这一步花了一整天排错；下面是提炼后的、已验证的路径。

已验证环境：

| 组件 | 版本 |
|---|---|
| GPU | Tesla V100-PCIe-32GB (sm_70)，驱动 580 |
| 操作系统 | Ubuntu 24.04 |
| CUDA toolkit (nvcc) | 12.8 (12.8.93)，与系统 CUDA 12.0 并存安装 |
| CMake | 3.28.3 |
| CUTLASS | 4.4.2（本地 tarball） |
| 库 | libcurl 8.5，ffmpeg 60.x（libavformat/libavcodec/libavutil/libswscale） |

## 太长不看版（TL;DR）

```bash
# 1. 系统软件包
sudo apt-get install -y build-essential cmake pkg-config \
  libcurl4-openssl-dev libavformat-dev libavcodec-dev \
  libavutil-dev libswscale-dev

# 2. CUDA toolkit 12.8+（nvcc）。我们系统的 toolkit 是 12.0；我们并行安装了
#    12.8（静默安装 toolkit 不动驱动），并让 CMake 显式指向它。
#    https://developer.nvidia.com/cuda-downloads

# 3. 源码
git clone https://github.com/liujun-7788/ninfer-v3-v100.git
cd ninfer-v3-v100

# 4. CUTLASS 4.4.2 放本地目录（见坑 2）
wget https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.4.2.tar.gz \
  -O cutlass-4.4.2.tar.gz
tar xzf cutlass-4.4.2.tar.gz
# 如果 tarball 解压出双层嵌套（cutlass-4.4.2/cutlass-4.4.2/），
# 把内层拍平上来，确保 ./cutlass-4.4.2/CMakeLists.txt 存在

# 5. 配置 + 编译（只编 sm_70——这是最大的提速点）
cmake -B build-v100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DFETCHCONTENT_SOURCE_DIR_CUTLASS=$PWD/cutlass-4.4.2
cmake --build build-v100 -j$(nproc)

# 产物：build-v100/apps/ninfer-serve
```

## 我们踩过的坑（一天时间花在哪）

1. **`CMAKE_CUDA_ARCHITECTURES=70`。** 默认配置会为一长串架构编译（70 到 121a）。显式传 `70`——V100 分支的内核本来就硬性检查计算能力 7.0（RTX 4090 是 sm_89，跑不了这个构建，启动时会被拒绝；不要试图用别的卡替代），编译速度也大幅加快。

2. **CUTLASS 走 FetchContent。** 如果你的网络到不了 GitHub（我们的生产机到不了），CMake 的 FetchContent 下载会挂起或失败。手动下载 v4.4.2 tarball，并把 `-DFETCHCONTENT_SOURCE_DIR_CUTLASS` 指向解压目录。注意：tag tarball 解压后可能**双层嵌套**（`cutlass-4.4.2/cutlass-4.4.2/`）——把内层拍平上来，确保 `<目录>/CMakeLists.txt` 存在，否则配置失败。

3. **CUDA toolkit 版本。** 我们用 toolkit 12.8（nvcc 12.8.93）验证通过，系统默认是 12.0。上游项目面向 Blackwell（sm_120a）并假定 12.8；与其和混合工具链搏斗，不如并行安装 12.8（不碰驱动），并传 `-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc`。

4. **链接阶段的 nvlink `Skipping incompatible ...` 警告**（libdl / librt / libpthread）无害——引擎链接与运行正常。不要去追它们。

5. **媒体前端。** 图像/视频输入需要 ffmpeg 开发库（libavformat/libavcodec/libavutil/libswscale）与 libcurl，配置阶段经 pkg-config 检测。想要文本之外的输入输出，请在 cmake 之前装好。

## V100 上重要的运行参数

我们每天运行的生产线（Qwen3.8-27B NVFP4，官方 v3 工件）：

```bash
build-v100/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 7105 --device 0 --model-id qwen3.8-27b \
  --max-context 131072 --kv-capacity auto --max-concurrency 1 \
  --kv-dtype int8 --device-state-slots 1 --host-state-slots 8 \
  --host-kv-mib 8192 --spec mtp --draft-tokens 3 --lm-head-draft \
  --preserve-thinking --pending-timeout-ms 600000
```

- **CUDA Graphs 在 V100 上可用**（这套设置下捕获约 0.3–0.7 秒）。如果捕获 OOM，说明你的 KV 预算对剩余显存太激进：先降 `--max-context`；万不得已才退到 `--no-cuda-graph`。
- **MTP 草稿 token 数（K）。** 我们在 Qwen3.8-27B NVFP4 上用真实 HTTP 负载扫了 K=2..5（上下文 2K/8K/32K/64K/128K，各 3 次重复，各 K 使用完全相同的提示）：

  | K | 平均解码 tok/s（5 档 ctx） | 2K | 8K | 32K | 64K | 128K | 接受率中位数 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | 2 | 78.2 | 80.7 | 119.6 | 85.1 | 70.6 | 35.2 | 71.3% |
  | **3** | **84.2** | **123.0** | **115.4** | 78.7 | 56.4 | **47.5** | 70.0% |
  | 4 | 61.7 | 73.3 | 75.5 | 73.6 | 54.8 | 31.1 | 42.9% |
  | 5 | 77.4 | 109.7 | 90.3 | 85.1 | 69.8 | 32.2 | 66.1% |

  **K=3 是综合最优默认值。** 64K 上下文时 K=2 胜出；K=4 崩塌（接受率跌到约 40–47%，吞吐随之下降）。Prefill 与 K 无关（每个上下文档位跨 K 波动 <1%）。

- **Prefill 是串行的。** 引擎一次只跑一个 prefill；并发请求会排队，若 prefill 超过 `--pending-timeout-ms`（默认 30 秒）将收到 HTTP 503。生产环境请调大（我们用 600000），或保持 `--max-concurrency 1`。
- **首次冷启动需要几分钟**（权重分页 + 图捕获）。用 HTTP 端点（`GET /v1/models`）探测就绪，不要看 systemd 状态——unit 可能显示 "active" 而引擎尚未监听。

## 模型

官方上游 v3 工件可用本分支直接加载，例如
[neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)。
v2 工件（magic `NInfer\0\2`）照常工作，无需任何改动。
