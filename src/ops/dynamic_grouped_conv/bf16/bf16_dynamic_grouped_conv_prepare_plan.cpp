#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"

#include "ops/dynamic_grouped_conv/volta/dflash2_dynamic_conv_volta.h"
#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden = 5120;
constexpr int kDeltaRows = 1280;

// sm_70 port: composed from RMSNorm + general BF16 linear (the W_delta projection) + a small
// depthwise-conv kernel. Scratch is the normalized input and the projected deltas.
std::size_t scratch_bytes(int columns) {
    return static_cast<std::size_t>((kHidden + kDeltaRows)) * columns * sizeof(std::uint16_t);
}

} // namespace

std::size_t bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(int min_width, int max_width,
                                                                       int min_batch,
                                                                       int max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument("dynamic grouped conv prepare workspace: invalid W/B interval");
    return scratch_bytes(max_width * max_batch);
}

void bf16_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm,
                                                float eps, const Tensor& base, const Weight& weight,
                                                Tensor& prepared, Tensor& finish,
                                                WorkspaceArena& workspace, cudaStream_t stream) {
    const int width   = residual.ne[1];
    const int batch   = residual.ne[2];
    const int columns = width * batch;

    auto scope    = workspace.scope();
    Tensor n      = workspace.alloc(DType::BF16, {kHidden, columns});
    Tensor delta  = workspace.alloc(DType::BF16, {kDeltaRows, columns});

    rmsnorm(residual.view({kHidden, columns}), norm, eps, false, n, stream);
    dflash2_bf16_control_projection_launch(
        static_cast<const __nv_bfloat16*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(n.data), kDeltaRows, kHidden, columns,
        static_cast<__nv_bfloat16*>(delta.data), stream);

    dflash2_dynamic_conv_prepare_launch(static_cast<const __nv_bfloat16*>(n.data),
                                        static_cast<const __nv_bfloat16*>(delta.data),
                                        static_cast<const __nv_bfloat16*>(base.data), width, batch,
                                        static_cast<__nv_bfloat16*>(prepared.data),
                                        static_cast<__nv_bfloat16*>(finish.data), stream);
}

} // namespace ninfer::ops::detail
