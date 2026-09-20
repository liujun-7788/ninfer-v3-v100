# PP2 双卡层流水线改造 — 实施计划（2026-09-21）

状态：通信基建已完成并验证（commit b6d022f8 TpGroup、de05f850 PpLink）。
本文是下一阶段的精确实施清单。工作目录 `/data/deploy/ninfer-mp`（mp-dev 分支），
本地镜像 `C:\Users\liujun\ninfer-mp-src`。

## 背景与决策

- 目标：预填充提速 ~1.8x（131k: 530→~1000 tok/s 追平 1catvllm 966；4 卡 PP4 ~2000）。
  解码保持单轮层切分串行（速度不变，PP 不影响）。
- TP 被搁置的原因：Program 句柄绑定 owner 指针（fan-out 代理需翻译全部复合类型）、
  NVFP4/FP8 QPN 预打包权重无法按列切片（需 repack）、48 个 GDN 层的 conv/recurrent
  状态切分。若后续仍要 TP，TpGroup 已就绪（解码图内 12.8µs/预填充 21MB 2.6ms）。
- PP2 利用现有 `--prefill-chunk 2048`：chunk 即微批次，并发 1 近零气泡。
  rank0 = 层 0-31 + embedding + vision；rank1 = 层 32-63 + final_norm + lm_head + MTP + 采样。

## 已验证事实

- 好卡对 GPU3↔GPU4（同 NUMA）：P2P 13GB/s；跨 NUMA 的 GPU1↔GPU2 allreduce 慢 3.5x。
- PCIe 上跨 GPU 自旋**设备内存** flag 不可靠（L2 可见性），必须用主机 pinned 邮箱。
- 单写者单调 flag 是硬约束：多块竞写不同代数到同一字会回退死锁（PpLink/TpGroup 已按此设计）。
- 设备侧计数器（atomicAdd）保证 CUDA Graph 重放安全。
- NCCL 单进程双 comm 需 group 语义且仍会挂；torch 多进程 NCCL 好卡对 20GB/s bus，
  但小消息 76µs 远差于自研 12.8µs → 不引入 NCCL。
- 服务器测试要 `stdbuf -o0` + `timeout`，否则 stdout 缓冲丢失难排障。

## 实施清单（按依赖顺序）

1. **镜像 KV 池**（src/core/paged_kv_cache.{h,cpp}）
   - `DeviceKVPagePool` 增加可选 `DeviceKVPagePool* mirror_`；构造时注入。
   - 所有变更操作（reserve/resize/materialize/materialize_one/dematerialize/
     dematerialize_one/zero_pages/copy_page/copy_to_host/copy_from_host 及内部
     release_*）在自身操作后以相同索引对 mirror 应用一次。两池内部自由表状态
     一致 ⇒ 相同选择 ⇒ 页号跨 rank 一致，block table 无需翻译。
   - `KVExecutionTablePool` 同样加 mirror（publish/publish_repeated/acquire/release_row）。
   - 注意 lease/Reservation 仍只属于主池；镜像池只做物理对齐。

2. **ExecutionCore 扩展**（src/targets/qwen3_6/impl/runtime/schedule.h）
   - 增加字段：`std::uint32_t layer_begin, layer_end;` `PpLink* pp;` `std::size_t pp_site;`
     `void* peer_hidden;`（对端 x 缓冲地址，主机侧每 unit 填写）。
   - 单卡路径全部字段为 {0, n_layers, nullptr} —— 零行为变化。

3. **TextContext 层分段**（text_context_impl.h）
   - `run_layers`：循环范围改 `[execution.layer_begin, execution.layer_end)`；
     入口 `layer_begin>0` 时 `pp->wait(rank, site)`；出口 `layer_end<n_layers` 时
     `pp->push(rank, x, peer_hidden, T*5120*2, site)`。
   - `sample_from_hidden`、MTP 相关路径：仅 `layer_end==n_layers` 的 rank 执行
     （program 侧已保证只在该 rank 调用）。
   - 注意 prefill_impl 里 final token 采样/MTP bridge 在 rank1。

4. **ProgramImplCore 双 rank 化**（program_impl.h）
   - 成员加 `std::optional<PpLink> pp_;`（PpLink 拥有两个 DeviceContext，
     原 `device` 引用绑 rank0）。`std::array<ExecBundle,2>`：{model 视图, work arena,
     linear_attention 池, io(RoundState), prefill_hidden, replay_records}。
   - KV 池/表池：rank0 主池 + rank1 镜像（几何按 32 层减半 → 每卡 KV 减半）。
   - 上下文构造（PrefillContext/OrdinaryBatchContext/MtpBatchContext）按 rank
     各构造一份；schedule 函数每 rank 调一次（锁步）。
   - prefill：每 chunk 先后 enqueue A(rank0)/B(rank1)，除 finalize 外不同步
     （跨 chunk 流水：B(i) 与 A(i+1) 天然并行，KV 写区间不相交，workspace 各 rank 独立）。
   - decode：每 rank 捕获各自的半图；rank0 图尾 push+signal，rank1 图首 wait。
     解码轮次闸门：rank1 图尾 ack-signal，rank0 下轮图首 ack-wait，
     ack 计数器 `arm_zero_wait` 初始化为 -1（首轮直通，避免循环等待）。
   - 采样/egress：只从 rank1 的 host 缓冲读取。

5. **加载分段**（targets/qwen3_6_27b/impl/load/bindings.cpp）
   - 绑定参数加 (rank, split)：rank0 只绑层 0-31 权重 + token_embedding + vision；
     rank1 只绑层 32-63 + final_norm + output_head + MTP + proposal。
     不在范围内的层保持空 Weight/Tensor（payload=null）——数组尺寸不变（编译期），
     TextContext 循环范围已保证不触碰。
   - 每 rank 在各自 device 上 materialize（两次读同一 artifact，简单可靠）。

6. **CLI/启动**（apps/serve/main.cpp、serve_options、engine.cpp、registry）
   - `--pp-devices A,B`；engine.cpp 构造两个 DeviceContext 交给 PpLink；
     Instance 构造把两个 rank 的束传入 Program。

7. **验证阶梯**
   a. 单卡回归：`--device 1` 行为与现有 build 完全一致（pp 字段为空）。
   b. PP2 正确性：同 prompt，单卡 vs `--pp-devices 3,4` 输出 token 逐个比对
      （贪心采样下应完全一致；数值差异仅来自边界——理论上无）。
   c. 基准：并发 1，1k/8k/32k/131k 提示长度，对比单卡与 1catvllm（7004）。
      注意长测前停 7004（占用 GPU3/4），测完恢复；7005/7006 用 GPU1/2 不受影响。

## 风险与备注

- `advance_prefill` 若在 chunk 间同步主机，先确认（现在证据：无，流式返回 PrefillProgress）。
- prefix 复用/checkpoint 捕获的 host 传输走主池 + 镜像自动覆盖。
- Vision 的 visual embeddings 在层 0 之前进入 x（rank0），PP 下无跨卡问题。
- DFlash 在 Volta 构建本就是 stub，无需处理。
- 测试脚本：tools/pp_link_smoke.cu、tools/tp_group_smoke.cu、p2p_test.cu 已在仓库根。
