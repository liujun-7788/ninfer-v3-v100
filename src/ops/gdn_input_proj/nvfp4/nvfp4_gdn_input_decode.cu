#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                   cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
        Nvfp4GdnInputOutput<Geometry>{static_cast<__nv_bfloat16*>(qkv.data),
                                      static_cast<__nv_bfloat16*>(z.data)});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    if (weight.n == Nvfp4GdnInputShardGeometry::kOutputRows) {
        launch_decode<Nvfp4GdnInputShardGeometry>(x, weight, qkv, z, stream);
        return;
    }
    launch_decode<Nvfp4GdnInputGeometry>(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
