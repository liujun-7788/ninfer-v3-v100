# PP2 双卡层流水线改造 — 实施计划（2026-09-21）

状态更新（同日晚）：
- ✅ 步骤 1（镜像池）完成并编译通过：commit 228bcbac。DeviceKVPagePool.set_mirror +
  mirror_take_pages/mirror_release_page（raw 索引路径，绕过 owner/代数校验），
  materialize/materialize_one/release_page 自动镜像；zero_pages/copy_page 按
  contiguous run 同时写主/镜像池平面；KVExecutionTablePool.set_mirror + publish_indices
  直写镜像矩阵。copy_to_host/from_host 故意不镜像（PP 模式禁用 host offload）。
- ✅ 步骤 2（ExecutionCore）+ 步骤 3（TextContext 层分段）完成：commit 0c1cb00c。
  ExecutionCore 新增 layer_begin/layer_end/gdn_offset/attn_offset/pp/pp_site/pp_rank/
  peer_hidden/peer_hidden_bytes（默认全零=单卡完整行为，所有旧调用点零改动）。
  TextContext::set_pipeline(Pipeline)；run_layers 入口 wait+boundary DtoD copy、
  出口 push；attn/gdn/replay 的索引平移；next_projection_hints 按 rank 边界截断；
  ordinary_decode_batch / target_verify_batch_impl / prefill_impl 的 stem（embedding）
  与 tail（rmsnorm/lm_head/sample/MTP-prep）按 first/last 门控。
- 全量构建 0 error，apps/ninfer-serve 正常产出。
- 提交链：b6d022f8 → de05f850 → c54186d9 → 228bcbac → 0c1cb00c。

## 剩余工作（步骤 4-7 的精确落点）

### 步骤 4：ProgramImplCore 双 rank 化（program_impl.h）
已确认的关键锚点：
- `execution_core` lambda 在 ~11182 行（warmup 图捕获区），构造 ExecutionCore 的唯一工厂；
  另有两处内联构造（11567 附近 prefill、11892/12091 附近 decode_raw/mtp_raw）。
- 物理存储构造在构造函数 833-960：`decoder = make_unique<DecoderState>(backing, plan.persistent.decoder)`
  → `text_kv`（PagedKVCache）、`state_images`（StateImageDevicePool）、replay records 等，
  全部从**同一布局计划**绑定到 backing DeviceSpan。
- **rank1 实例化模式（本轮定稿）**：rank1 的 DecoderState/StateImageDevicePool/replay
  records/work 各自在 rank1 设备上 cudaMalloc 一块同尺寸 backing，用**同一个 plan 子布局**
  再构造一份（布局计划单拷贝，物理双份）；rank1 的 KV 池/表池与 rank0 的 set_mirror 互连。
  PagedKVCacheView 是 (cache指针, block_table张量) 绑定 —— 每 rank 用自己的 cache 构建
  视图，页号经镜像天然一致，表内容经 publish 镜像天然一致。
- weights：rank1 的 LoadedModelData 由加载路径第二遍产出（见步骤 5）。
- `backend_kv_cache()` 10776、`text_kv_view/mtp_kv_view` 10954。
改造内容：
1. PpLink 实例（拥有两个 DeviceContext；原 `device` 成员绑 rank0）。
2. `RankBundle`：{DeviceContext*, LoadedModelData*, WorkspaceArena*, LinearAttentionStatePool*,
   RoundState*(io1), Tensor* prefill_hidden1, PagedKVCache*(rank1 cache), 图族副本,
   ordinary/mtp host 缓冲副本, boundary Tensor(21MB cudaMalloc, 启动固定)}。
3. rank0 的 KV 池为主、rank1 为 mirror（set_mirror 互指：主池→镜像池单向即可，
   镜像操作全部由主池发起）；几何按各自 32 层减半；表池同构镜像。
4. 图捕获：per-rank 调 capture_*（每 rank 一个 OrdinaryBatchContext/MtpBatchContext）。
   解码轮次闸门：rank1 图尾 ack-signal(site_ack)，rank0 图首 ack-wait，
   `arm_zero_wait(0, site_ack)` 初始化 -1（首轮直通）。PpLink 需补一个
   `signal(consumer→producer 反向)` 便捷接口（现 push 是单向拷贝+signal，
   ack 只需 signal 不拷贝——给 PpLink 加 `signal_only(rank, site)`）。
5. decode_raw / advance_prefill_raw / mtp 路径：上下文按 rank 构造、schedule 函数
   每 rank 调一次；采样/egress 只读 rank1。
6. prefill 流水：enqueue 顺序 A(i)@rank0 → B(i)@rank1（B 内部 wait），除 finalize 外
   主机不同步。跨 chunk 安全性已论证（KV 区间不相交、workspace 各 rank 独立、
   boundary 地址启动固定）。
7. V1 约束：--pp-devices 与 host-offload/context-cache/vision 互斥（启动时报错），
   max-concurrency 不变（引擎逻辑不动，PP 只影响 Program 内部）。

### 步骤 5：加载分段（bindings.cpp / package.cpp）
- LoadedModel 数组尺寸不变（编译期 16/48）；rank 只 materialize 自己层范围的张量，
  其余层留 null view。embedding/vision → rank0；final_norm/output_head/MTP/proposal → rank1。
- 在 package 构造处按 rank 切两次 device 上下文各加载一遍 artifact。

### 步骤 6：CLI
- serve_options/parse：`--pp-devices A,B`；engine.cpp：initialize_device 分支；
  registry/construct_target 传递设备对。

### 步骤 7：验证（用户已授权随时停 7004/7005/7006，测完恢复）
a. 单卡回归：`--device 1` 起 7005 同配置，冒烟对话。
b. PP2 正确性：`--pp-devices 3,4` 同 prompt 贪心解码 vs 单卡逐 token 比对。
c. 基准：并发 1，1k/8k/32k/131k，对比 1catvllm 表格指标（TTFT/预填充/输出速度）。
   测前停 7004（占 GPU3/4）+ 7005/7006（GPU1/2 可做单卡对照组），测完恢复。

## 风险与备注

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
