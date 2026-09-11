#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Fused plain RMSNorm + full-128 split-half 1-D RoPE, in place. x is BF16 [128, heads, T],
// weight BF16 [128], positions I32 [T]. n stays FP32 through the rotation.
void dflash2_rmsnorm_rope_launch(const Tensor& x, const Tensor& weight, const Tensor& positions,
                                 float eps, float theta, cudaStream_t stream);

} // namespace ninfer::ops::detail
