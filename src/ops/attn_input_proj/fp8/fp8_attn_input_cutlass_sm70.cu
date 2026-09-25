#include "ops/attn_input_proj/fp8/fp8_attn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Allocator>
Tensor allocate_projected(Allocator& allocator, int rows, int tokens) {
    return allocator.alloc(DType::BF16, {rows, tokens});
}

__global__ void split_attn_output(const __nv_bfloat16* __restrict__ projected,
                                  __nv_bfloat16* __restrict__ q,
                                  __nv_bfloat16* __restrict__ gate,
                                  __nv_bfloat16* __restrict__ k,
                                  __nv_bfloat16* __restrict__ v, std::int32_t rows,
                                  std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const Fp8AttentionInputOutput output{q, k, gate, v, rows};
    const int row   = static_cast<int>(i % rows);
    const int token = static_cast<int>(i / rows);
    *output.destination(row, token) = projected[i];
}

} // namespace

std::size_t fp8_attn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected(layout, Fp8AttnInputGeometry::kOutputRows, tokens);
    return layout.peak_bytes(1) +
           fp8_cutlass_sm70_workspace_bytes(Fp8AttnInputGeometry::kOutputRows,
                                            Fp8AttnInputGeometry::kInputRows, tokens);
}

void fp8_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                        Tensor& gate, Tensor& k, Tensor& v,
                                        WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    const int rows   = weight.n;
    Tensor projected = allocate_projected(workspace, rows, x.ne[1]);
    fp8_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * rows;
    split_attn_output<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(v.data), rows, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
