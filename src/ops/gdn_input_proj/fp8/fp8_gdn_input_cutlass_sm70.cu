#include "ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_cutlass_sm70.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Allocator>
Tensor allocate_projected(Allocator& allocator, int rows, int tokens) {
    return allocator.alloc(DType::BF16, {rows, tokens});
}

__global__ void split_gdn_output(const __nv_bfloat16* __restrict__ projected,
                                 __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ z,
                                 std::int32_t rows, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const std::int32_t qkv_rows = rows * 5 / 8;
    const std::int32_t z_rows   = rows * 3 / 8;
    const int row   = static_cast<int>(i % rows);
    const int token = static_cast<int>(i / rows);
    if (row < qkv_rows) {
        qkv[static_cast<std::int64_t>(token) * qkv_rows + row] = projected[i];
    } else {
        z[static_cast<std::int64_t>(token) * z_rows + row - qkv_rows] = projected[i];
    }
}

} // namespace

std::size_t fp8_gdn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected(layout, Fp8GdnInputGeometry::kOutputRows, tokens);
    const std::size_t projected_bytes = layout.peak_bytes(1);
    return projected_bytes + fp8_cutlass_sm70_workspace_bytes(
                                 Fp8GdnInputGeometry::kOutputRows, Fp8GdnInputGeometry::kInputRows,
                                 tokens);
}

void fp8_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    const int rows   = weight.n;
    Tensor projected = allocate_projected(workspace, rows, x.ne[1]);
    fp8_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * rows;
    split_gdn_output<<<static_cast<int>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data), static_cast<__nv_bfloat16*>(qkv.data),
        static_cast<__nv_bfloat16*>(z.data), rows, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
