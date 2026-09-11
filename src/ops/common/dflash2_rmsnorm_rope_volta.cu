// DFlash2 fused plain-RMSNorm + full-128 split-half 1-D RoPE (theta 1e7) — sm_70.
//
// n is kept in FP32 through the rotation; only the final rotated result is rounded to BF16 once.
// The angle uses the fork's precise double-precision inv-frequency table + range reduction.

#include "ops/common/dflash2_rmsnorm_rope_volta.h"

#include "core/device.h"
#include "ops/common/dflash_rope.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim = 128;
constexpr int kHalf    = 64;

__global__ void dflash2_rmsnorm_rope_kernel(__nv_bfloat16* __restrict__ x,
                                            const __nv_bfloat16* __restrict__ weight,
                                            const std::int32_t* __restrict__ positions, int heads,
                                            int tokens, float eps) {
    // x is [128, heads, T] contiguous, so column col = head + heads * token.
    const int col = static_cast<int>(blockIdx.x);
    const int tok = col / heads;
    if (tok >= tokens) { return; }
    const int d = static_cast<int>(threadIdx.x); // 0..127

    __nv_bfloat16* row = x + static_cast<std::int64_t>(col) * kHeadDim;
    const float xd     = __bfloat162float(row[d]);

    __shared__ float sums[4];
    const float ss = block_reduce_sum<128>(xd * xd, sums);
    __shared__ float inv_norm;
    if (d == 0) { inv_norm = rsqrtf(ss / static_cast<float>(kHeadDim) + eps); }
    __syncthreads();

    __shared__ float n[kHeadDim];
    n[d] = xd * inv_norm * __bfloat162float(weight[d]);
    __syncthreads();

    if (d < kHalf) {
        const float lo = n[d];
        const float hi = n[d + kHalf];
        float s;
        float c;
        ninfer::ops::dflash_rope_sincos(positions, tok, d, &s, &c);
        row[d]         = __float2bfloat16_rn(lo * c - hi * s);
        row[d + kHalf] = __float2bfloat16_rn(hi * c + lo * s);
    }
}

} // namespace

void dflash2_rmsnorm_rope_launch(const Tensor& x, const Tensor& weight, const Tensor& positions,
                                 float eps, float theta, cudaStream_t stream) {
    (void)theta; // the inv-frequency table is fixed for theta = 1e7
    const int heads  = x.ne[1];
    const int tokens = x.ne[2];
    const int blocks = heads * tokens;
    dflash2_rmsnorm_rope_kernel<<<blocks, kHeadDim, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const std::int32_t*>(positions.data), heads, tokens, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
