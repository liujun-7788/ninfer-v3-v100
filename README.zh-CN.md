[English](README.md) | **简体中文**

> **ninfer-v3-v100** — [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)（即 [Neroued/ninfer](https://github.com/Neroued/ninfer) 的 Tesla V100 分支）的持续维护分支，新增**对官方 v3 `.ninfer` 工件的直接读取支持**——官方下载的 [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) 等工件无需转换即可直接运行。经验证的 V100 编译指南、运行参数与 MTP 基准数据见 **[V100 编译与运行指南（中文版）](docs/V100-BUILD.zh-CN.md)**（[英文版](docs/V100-BUILD.md)）。本分支与上游作者无隶属或背书关系。

# NInfer

> 单张 V100 上，Qwen 3.8 27B 解码速度最高 219 tok/s——Volta 上的软件 NVFP4。

NInfer 是一个从零编写的 C++/CUDA 推理引擎，针对 NVIDIA Tesla V100 上的精选 Qwen 检查点做了优化。

它通过本地 CLI 或兼容 OpenAI-/Anthropic 的 HTTP API 支持文本、图像和视频输入。运行时定位刻意收窄：单 GPU、单常驻模型、1–8 个活跃请求。

## 模型

| 模型 | 权重 | 工件 | 下载与模型卡 |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |

工件内含精确的模型权重、分词器、聊天模板以及所需的媒体前端资源。

## 性能

Qwen3.8-27B NVFP4 在 K=1 时达到 **218.98 解码 tok/s**，MTP 草稿接受率 99.2%。
该结果来自 V100-PCIe-32GB（PCIe 版，非 SXM）。对应的 SXM2 显卡大约快 7%；解码主要受 HBM 带宽限制，PCIe 带宽与主机性能影响甚微。

### Tesla V100：软件 NVFP4 与 groupwise 推理

Qwen3.8-27B NVFP4 短上下文目标轮次（在下述 width-6+ 验证修复之后重测），K=1 草稿 token 时达到峰值：**219.0 解码 tok/s**，草稿接受率 99.2%——在该语料上窄窗口直接胜出；完整 K 扫描见下表。

单请求扫描使用公开 Engine 基准，硬件为 Tesla V100-PCIe-32GB，CUDA 12.8、INT8 group-64 KV。Prefill 为独立的 `pp2048` 运行；decode 为 `pp2048+tg256`，启用 CUDA Graphs 与优化后的提议头。每组结果使用 1 次丢弃的预热和 3 次计量重复。在 V100 笔记的 DFlash 窗口扫描中，性能更好的 V100-SXM2-32GB 每轮约快 7%；该解码负载受 HBM 限制，主机与 PCIe 总线几乎无影响。

| 模型配置 | K | Prefill tok/s | Decode tok/s | 草稿接受率 |
|---|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` MTP | 4 | 1,085.0 | 54.54 | 66.5% |
| Qwen3.6-27B `nvfp4` MTP | 5 | 223.8 | 55.22 | 54.4% |
| Qwen3.8-27B `groupwise-int` MTP | 5 | 1,083.9 | 130.96 | 97.1% |
| Qwen3.8-27B `nvfp4` MTP | 5 | 1,100.3 | 199.58 | 97.1% |
| Qwen3.8-27B `groupwise-int` DFlash2 | 7 | 1,044.2 | 77.84 | 100% |
| Qwen3.8-27B `nvfp4` DFlash2 | 7 | 1,059.0 | 126.32 | 100% |
| Qwen3.6-35B-A3B `groupwise-int` DFlash | 4 | 686.2 | 139.58 | 90.9% |

完整 Qwen3.8-27B `nvfp4` MTP 草稿窗口扫描（同一语料延续形态，width-6+ 修复移除了 sm_70 在 4 的上限之后）：

| K | Prefill tok/s | Decode tok/s | 草稿接受率 |
|---:|---:|---:|---:|
| 1 | 1,102.5 | **218.98** | 99.2% |
| 2 | 1,097.4 | 213.99 | 98.3% |
| 3 | 1,094.9 | 209.24 | 97.5% |
| 4 | 1,096.6 | 204.02 | 97.9% |
| 5 | 1,100.3 | 199.58 | 97.1% |
| 6 | 1,089.6 | 178.47 | 92.5% |
| 7 | 1,087.2 | 180.74 | 93.3% |

该语料是确定性延续，每个 K 的接受率都异常高、接近天花板，因此窄窗口直接胜出：一旦没有更多可接受的长度可买，每轮验证成本便占主导。请把它当作合成语料的上限，而非通用生成速率——在真实的、更不可预测的文本上，实际生产最佳点是 K=3（见本仓库历史中的长上下文扫描）。

解码吞吐强烈依赖草稿接受率——影响幅度见上文 MTP 扫描。sm_70 width-6+ 目标验证回归已修复（见下），因此 `--spec mtp` 现在接受与上游相同的 [1,7] 窗口，不再有 Volta 特有的上限。`--spec dflash2` 在同一语料延续形态下峰值在 K=7（3–10 扫描两侧均回落）；在所有尝试过的 K 上，MTP 仍然领先 DFlash2。

Qwen3.8-27B 工件还捆绑了 DFlash2——上游的掩码块投机解码器（`--spec dflash2`）。MTP 仍是 Volta 上通用解码的推荐后端。在一个多样、非重复的语料上，DFlash2 K=7 在 2K 和 32K 上下文以微弱优势击败了最佳 MTP 窗口，而 MTP 在 8K、16K、150K 领先。DFlash2 K=7 是其最强的静态默认值；基于接受率的窗口自适应在困难延续上可能偏向 K=3。完整对比见 [V100 移植笔记](docs/v100.md#varied-context-dflash2-sweep)。

当生成的后缀与之前某个 16-token 片段完全匹配、且学习到的提议头与查找延续一致时，MTP 会自动把验证扩展到最多 15 个草稿 token。在一个 172-token 逐字复制提示上，Qwen3.8-27B NVFP4 以 **201.0 tok/s** 输出完全一致的延续，平均每轮 12.91 个输出 token。这是上下文复现的快速路径；普通生成继续使用常规 MTP 窗口与上述通用解码结果。

上下文查找 MTP 受 [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090) 启发。

在 Volta 上，NVFP4 以软件方式解码执行，使用调优过的 FP16 tensor core 与 SIMT 内核。稠密 MLP gate/up 载荷在模型加载时于 GPU 上原位预打包为 QPN 解码布局；磁盘上的工件不变，推理不做运行时权重重排。Volta QPN 预打包工作受 [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) 启发。

35B-A3B 生产 DFlash 轮次在 2,048-token 上下文使用 K=3：两次预热后十次计量轮平均 **125.9 tok/s**、每轮平均 3.8 个输出 token。

## 快速开始

> 中文读者建议直接从 **[V100 编译与运行指南（中文版）](docs/V100-BUILD.zh-CN.md)** 开始——那是我们实际生产环境逐条验证过的编译路径和避坑清单。

NInfer 需要 64 位 Linux、Tesla V100 与 CUDA Toolkit 12.8、CMake 3.28 或更新、C++20 主机编译器、Ninja、`pkg-config`、FFmpeg 开发库（`libavformat >= 60`、`libavcodec >= 60`、`libavutil >= 58`、`libswscale >= 7`）以及 `libcurl >= 7.85`。本移植以 `sm_70` 编译。

显式选择 CUDA 12.8 与 Volta：

```bash
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j
```

相同的五个 `.ninfer` 工件使用公开的 Engine/CLI/serving 路由。资格验证与首选 GPU 启动器见 [V100 移植笔记](docs/v100.md)。

在 Volta 上加载 NVFP4 工件时，每个稠密 MLP gate/up 与 down 载荷会在 GPU 上原位重排为解码内核使用的 QPN 片段顺序。行缩放的 FP8 载荷同样被置换为其原生 Volta QPN 流。`.ninfer` 文件与主机侧工件字节不变，设备分配大小不变，临时重排存储在推理前释放。因此这些设备驻留权重在加载时被有意修改；V100 路径不为这些载荷保留检查点原生不可变布局。

默认构建排除测试、基准与维护者工具。没有 install 目标或打包的二进制分发；请从源码构建树运行 NInfer。

用 Hugging Face CLI 下载本示例使用的工件：

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

启动一个长驻文本/视觉 agent 服务器，单活跃请求通道 + Device/Host 检查点保留：

```bash
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --max-context 240000 \
  --prefill-chunk 2048 \
  --kv-capacity auto \
  --max-concurrency 1 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 4 \
  --lm-head-draft \
  --preserve-thinking \
  --vision
```

每个请求有 240,000-token 逻辑上限。Device KV 池按权重、工作区、状态、Vision 与图分配之后剩余的显存计算。缓存层提供 1 个 Device 检查点槽、8 个锁页 Host State 槽与 8 GiB 锁页 Host KV。

发送一个 OpenAI 风格的请求：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

运行一次性 CLI 请求，分配 32,768 token：

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode, then give a concise conclusion." \
  --max-context 32768 \
  --max-new 8192 \
  --kv-dtype fp8 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

回答内容写入 stdout。结构化的启动/运行时错误记录，以及 CLI 拥有的推理、计时、吞吐、内存与投机解码报告写入 stderr；推理与结果报告保持无前缀的产品输出。终端上，权重物化还会使用一行临时进度。重定向的 stderr 只接收持久结构化阶段记录，包括长加载的限速进度。选项与本地输入错误仍是直接的命令诊断。使用 `--messages FILE` 与 `--vision` 处理结构化图像/视频输入；见 [CLI 指南](docs/cli.md) 与[已提交示例](examples/cli/)。

## 资源感知的长上下文复用

可复用的前缀检查点包含其精确提示前沿的 KV 与完整续写状态。Device 驻留检查点可直接恢复。压力之下，规划器会权衡 Device 保留、锁页 Host State/KV，并按即时恢复工作量与后续复用成本进行驱逐。活跃请求保留其完成预留。

算法见 [资源调度与上下文缓存](docs/maintainer/resource-scheduling-and-context-cache.md)；热复用、Host 恢复、驱逐、共享前缀、调度边界与多模态负载的公开 HTTP 覆盖见 [Serve TTFT 基准](tools/bench/ttft/)。

## 评测

能力分数通过 NInfer 的 OpenAI 兼容 serving 路由测得，开启 thinking、MTP3、EvalScope 1.9.0（0-shot、规则评分、每题单样本）：

| 模型配置 | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

Qwen3.6 行使用 temperature 0.6、presence penalty 1.0；Qwen3.8 行使用 temperature 1.0、presence penalty 0.0。多模态评测使用 `--vision` 与 81,920-token 上下文上限。文本评测使用 262,144 token，Qwen3.8-27B NVFP4 除外——它使用 252,928 token 以匹配其实测内存包络。每个分数为每题单样本；模型卡含正确/总题数与评测说明。

## 启动说明

GPU 驻留在进程启动时固定。`--spec` 选择投机解码驻留，`--vision` 选择 Vision 驻留。DFlash 可用于纯文本 Qwen3.6-35B-A3B 执行，DFlash2 用于 Qwen3.8-27B。

## Docker

在装有 NVIDIA Container Toolkit 的主机上构建运行时镜像：

```bash
docker build --tag ninfer:local .
```

挂载下载好的模型并运行相同的示例服务器配置：

```bash
docker run --rm \
  --gpus '"device=0"' \
  --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" \
  ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 \
  --max-context 240000 \
  --kv-capacity auto \
  --max-concurrency 1 \
  --kv-dtype int8 \
  --device-state-slots 1 \
  --host-state-slots 8 \
  --host-kv-mib 8192 \
  --spec mtp --draft-tokens 4 \
  --lm-head-draft \
  --preserve-thinking \
  --vision
```

## 能力与边界

所有已注册模型 ID 支持：

- 带思考与非思考提示模式的文本生成；
- 图像、多图、视频与混合多模态消息；
- 分块 prefill、精确批 CUDA Graph 解码与启动期定界的批量解码；
- MTP 投机解码，草稿窗口 1–7；
- Qwen3.8-27B 的 DFlash2 掩码块投机解码（来自上游集成），草稿窗口 1–15；
- BF16、INT8、FP8 KV 存储；
- 离线因果困惑度评分；
- 带 Device/Host State 与 KV 保留的私有和共享精确前缀复用；
- 模型感知的采样默认值与显式采样器覆盖；
- OpenAI Responses Core、OpenAI Chat Completions 与 Anthropic Messages，包括流式、工具、本地响应状态、token 计数与用量核算。

35B-A3B 目标额外支持纯文本 DFlash，草稿窗口 1–15。

产品边界刻意保持很小：

- 每个 Engine 一张 Tesla V100、一个常驻模型；
- 启动期固定 1–8 个活跃请求容量，带定界的 FIFO 入口；
- 无请求抢占、优先级/QoS、活跃请求换入换出、权重卸载、多 GPU 或分布式 serving；
- 活跃请求与保留前缀共享一个启动期固定的 KV 池；
- 无运行时模型发现或未注册检查点回退；
- 解析出的工具调用返回给客户端；NInfer 不执行工具；
- 树内 C++ 头文件不作为已安装 SDK 分发。

`--max-context` 是每条序列的逻辑上限。`--kv-capacity` 为活跃请求与保留前缀共用的主文本 KV 池定容；`auto` 在启动时按权重之后剩余显存解析最大合法容量，并保留 1 GiB 定容余量。显式容量在进程生命周期内固定。

## 文档

- [文档索引](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [V100 资格验证与性能](docs/v100.md)
- [困惑度评测](docs/perplexity.md)
- [资源调度与上下文缓存](docs/maintainer/resource-scheduling-and-context-cache.md)
- [Serve TTFT 基准](tools/bench/ttft/)
- [CLI 示例](examples/cli/)
- [贡献指南](CONTRIBUTING.md)

运行相应的 `--help` 查看当前确切的选项契约。

## 许可证

NInfer 以 [Apache License 2.0](LICENSE) 授权。

已发布的工件派生自
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B)、
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 与
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)。Qwen3.6-27B NVFP4 工件
还使用了 [rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm)
的固定打包权重。Qwen3.8-27B NVFP4 工件还使用了
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) 的固定混合 FP8/NVFP4 权重。这些源
仓库以 Apache-2.0 分发。内嵌依赖在 `third_party/` 下保留各自的许可文件。
