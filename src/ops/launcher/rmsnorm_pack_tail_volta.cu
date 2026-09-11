// DFlash2 rmsnorm_pack_tail — sm_70.
//
// Plain multiplicative RMSNorm (eps 1e-6) applied to the non-anchor rows of each request block
// [5120,W,B], packed dense as [5120,(W-1)B]. Anchor row i=0 is never read.

#include "ops/launcher/rmsnorm_pack_tail.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden = 5120;

__global__ void rmsnorm_pack_tail_kernel(const __nv_bfloat16* __restrict__ input,
                                         const __nv_bfloat16* __restrict__ weight, int width,
                                         __nv_bfloat16* __restrict__ output) {
    const int packed_col = static_cast<int>(blockIdx.x);
    const int per_req    = width - 1;
    const int b          = packed_col / per_req;
    const int i          = packed_col % per_req + 1; // skip the anchor
    const int src_col    = i + width * b;

    const __nv_bfloat16* xcol = input + static_cast<std::int64_t>(src_col) * kHidden;
    __nv_bfloat16* ocol       = output + static_cast<std::int64_t>(packed_col) * kHidden;

    float ss = 0.0f;
    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        const float v = __bfloat162float(xcol[d]);
        ss += v * v;
    }
    __shared__ float sums[8];
    ss = block_reduce_sum<256>(ss, sums);
    __shared__ float inv_scale;
    if (threadIdx.x == 0) { inv_scale = rsqrtf(ss / static_cast<float>(kHidden) + 1.0e-6F); }
    __syncthreads();
    const float inv = inv_scale;

    for (int d = static_cast<int>(threadIdx.x); d < kHidden; d += static_cast<int>(blockDim.x)) {
        ocol[d] = __float2bfloat16_rn(__bfloat162float(xcol[d]) * __bfloat162float(weight[d]) * inv);
    }
}

} // namespace

void rmsnorm_pack_tail_launch(const Tensor& input, const Tensor& weight, Tensor& output,
                              cudaStream_t stream) {
    const int width = input.ne[1];
    const int batch = input.ne[2];
    const int cols  = (width - 1) * batch;
    rmsnorm_pack_tail_kernel<<<cols, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(weight.data), width,
        static_cast<__nv_bfloat16*>(output.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
