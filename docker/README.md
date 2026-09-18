# Docker deployment (Tesla V100 / sm_70)

[English | 简体中文](#简体中文)

## Prerequisites

- NVIDIA driver **>= 570** (the image ships CUDA 12.8 user-space libraries)
- Docker + GPU passthrough:
  - Linux: `nvidia-container-toolkit` (CDI or `--gpus` both work)
  - Windows: **Docker Desktop** with the WSL2 backend — same commands, one-click
- GPU: **Tesla V100 (sm_70) only.** The binary hard-requires compute capability
  7.0; consumer cards (RTX 4090 etc.) are rejected at startup. A 32 GB card
  fits the default recipe (230k context); smaller cards must lower
  `MAX_CONTEXT`.

## Step 1 — get the prebuilt binary

Download `ninfer-serve-sm70.tar.gz` from the
[Releases](../../releases) page, then unpack it next to the Dockerfile:

```bash
mkdir -p bin
tar xzf ninfer-serve-sm70.tar.gz -C bin    # produces bin/ninfer-serve
```

(In China, prefix the release asset URL with `https://ghfast.top/` if GitHub
downloads are slow.)

## Step 2 — build the image

```bash
docker build -t ninfer-v100 .
```

The base image (`nvidia/cuda:12.8.1-runtime-ubuntu24.04`, ~3 GB) is pulled
from Docker Hub on first build.

## Step 3 — run

Option A, the model is already on disk (mount it):

```bash
docker run -d --name ninfer-v100 \
  --gpus all \
  -p 7106:7106 \
  -v /path/to/models:/models \
  -e API_KEY=sk-change-me \
  ninfer-v100:latest
```

Option B, let the container download the model (~23.7 GB, resumable):

```bash
docker run -d --name ninfer-v100 \
  --gpus all \
  -p 7106:7106 \
  -v /path/to/models:/models \
  -e API_KEY=sk-change-me \
  -e AUTO_DOWNLOAD=1 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  ninfer-v100:latest
```

Or use Compose (`docker compose up -d`), see `docker-compose.yml`.

## Verify

```bash
curl -s http://127.0.0.1:7106/v1/models
```

Cold start takes ~1 minute (weight paging + CUDA graph capture). Check logs
with `docker logs -f ninfer-v100`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `API_KEY` | *(unset = no auth)* | Bearer token for the HTTP API |
| `MODEL_PATH` | `/models/qwen3_8_27b_nvfp4.ninfer` | Model artifact path inside the container |
| `MODEL_ID` | `qwen3.8-27b` | `model` field expected by API clients |
| `PORT` | `7106` | HTTP port inside the container |
| `DEVICE` | `0` | GPU index inside the container (use `NVIDIA_VISIBLE_DEVICES` / CDI to pick host GPUs) |
| `MAX_CONTEXT` | `230000` | Max context tokens; **lower (e.g. 200000) if startup planning fails on your card** |
| `KV_DTYPE` | `int8` | `int8` (recommended) or `fp8` (bigger pool but slower decode on Volta — measured) |
| `MAX_CONCURRENCY` | `2` | Concurrent decode slots (also sets `--device-state-slots`) |
| `SPEC` | `mtp` | Speculative decoding; `none` disables |
| `DRAFT_TOKENS` | `3` | MTP draft length (K=3 is the measured optimum) |
| `PREFILL_CHUNK` | `2048` | Prefill chunk size |
| `PENDING_TIMEOUT_MS` | `600000` | Queue timeout (10 min) before HTTP 503 |
| `VISION` | `1` | Enable image/video input (needs ~1 GB VRAM) |
| `HOST_STATE_SLOTS` / `HOST_KV_MIB` | `8` / `8192` | Host-side state/KV offload |
| `AUTO_DOWNLOAD` | `0` | `1` = download the model on first start |
| `HF_ENDPOINT` | `https://huggingface.co` | Mirror for the model download |

## Notes

- `--gpus all` passes every GPU through; to pin one card use
  `NVIDIA_VISIBLE_DEVICES=<index>` or CDI (`--device nvidia.com/gpu=2`).
- On Windows (Docker Desktop/WSL2) everything above is identical; GPU
  passthrough overhead is negligible and VRAM is not reduced.
- The official v3 artifact is
  [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
  (`qwen3_8_27b_nvfp4.ninfer`, ~23.7 GB). v2 artifacts also load unchanged.

---

# 简体中文

## 前提条件

- NVIDIA 驱动 **>= 570**（镜像内置 CUDA 12.8 用户态库）
- Docker + GPU 直通：
  - Linux：`nvidia-container-toolkit`
  - Windows：**Docker Desktop**（WSL2 后端）——命令完全相同
- 显卡：**仅支持 Tesla V100 (sm_70)**。二进制硬性要求计算能力 7.0，消费卡
  （RTX 4090 等）启动即被拒。32G 卡跑默认配方（230k 上下文）；更小的卡请调低
  `MAX_CONTEXT`。

## 第 1 步 — 获取预编译二进制

从本仓库 [Releases](../../releases) 页下载 `ninfer-serve-sm70.tar.gz`，解压到
Dockerfile 旁边：

```bash
mkdir -p bin
tar xzf ninfer-serve-sm70.tar.gz -C bin    # 得到 bin/ninfer-serve
```

（国内下载慢可在资产 URL 前加 `https://ghfast.top/` 前缀。）

## 第 2 步 — 构建镜像

```bash
docker build -t ninfer-v100 .
```

首次构建会从 Docker Hub 拉取基础镜像（`nvidia/cuda:12.8.1-runtime-ubuntu24.04`，
约 3 GB）。

## 第 3 步 — 运行

方式 A，模型已在本地（挂载进去）：

```bash
docker run -d --name ninfer-v100 \
  --gpus all \
  -p 7106:7106 \
  -v /path/to/models:/models \
  -e API_KEY=sk-change-me \
  ninfer-v100:latest
```

方式 B，由容器自动下载模型（约 23.7 GB，支持断点续传）：

```bash
docker run -d --name ninfer-v100 \
  --gpus all \
  -p 7106:7106 \
  -v /path/to/models:/models \
  -e API_KEY=sk-change-me \
  -e AUTO_DOWNLOAD=1 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  ninfer-v100:latest
```

也可以用 Compose（`docker compose up -d`），见 `docker-compose.yml`。

## 验证

```bash
curl -s http://127.0.0.1:7106/v1/models
```

冷启动约 1 分钟（权重分页 + CUDA 图捕获）。日志：`docker logs -f ninfer-v100`。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `API_KEY` | *(不设 = 无鉴权)* | HTTP API 的 Bearer token |
| `MODEL_PATH` | `/models/qwen3_8_27b_nvfp4.ninfer` | 容器内模型工件路径 |
| `MODEL_ID` | `qwen3.8-27b` | 客户端请求里的 `model` 字段 |
| `PORT` | `7106` | 容器内 HTTP 端口 |
| `DEVICE` | `0` | 容器内 GPU 编号（用 `NVIDIA_VISIBLE_DEVICES` / CDI 选宿主机卡） |
| `MAX_CONTEXT` | `230000` | 最大上下文 token；**启动规划失败就调低（如 200000）** |
| `KV_DTYPE` | `int8` | `int8`（推荐）或 `fp8`（池更大但 Volta 实测 decode 更慢） |
| `MAX_CONCURRENCY` | `2` | 并发解码槽数（同时设定 `--device-state-slots`） |
| `SPEC` | `mtp` | 投机解码；`none` 关闭 |
| `DRAFT_TOKENS` | `3` | MTP 草稿长度（实测 K=3 最优） |
| `PREFILL_CHUNK` | `2048` | prefill 分块 |
| `PENDING_TIMEOUT_MS` | `600000` | 排队超时（10 分钟）后返回 HTTP 503 |
| `VISION` | `1` | 开启图像/视频输入（约多占 1 GB 显存） |
| `HOST_STATE_SLOTS` / `HOST_KV_MIB` | `8` / `8192` | 主机侧状态/KV 卸载 |
| `AUTO_DOWNLOAD` | `0` | `1` = 首次启动自动下载模型 |
| `HF_ENDPOINT` | `https://huggingface.co` | 模型下载镜像源 |

## 备注

- `--gpus all` 会透传所有卡；只给一张卡用 `NVIDIA_VISIBLE_DEVICES=<编号>`
  或 CDI（`--device nvidia.com/gpu=2`）。
- Windows（Docker Desktop/WSL2）下以上命令完全一致；GPU 直通开销可忽略，
  显存无损耗。
- 官方 v3 工件 =
  [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)
  （`qwen3_8_27b_nvfp4.ninfer`，约 23.7 GB）。v2 工件同样可以直接加载。
