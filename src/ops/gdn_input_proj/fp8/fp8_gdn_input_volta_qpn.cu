#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

// See the attention sibling: same kernel, same reason, different epilogue.
void launch_fp8_gdn_input_volta_qpn(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    const void* x_fp16, cudaStream_t stream) {
    auto make_output = [&](auto geometry_tag) {
        using Geometry = decltype(geometry_tag);
        return Fp8GdnInputOutput<Geometry>{static_cast<__nv_bfloat16*>(qkv.data),
                                           static_cast<__nv_bfloat16*>(z.data)};
    };
    if (x_fp16 != nullptr) {
        if (weight.n == Fp8GdnInputShardGeometry::kOutputRows) {
            launch_fp8_volta_qpn_with_fp16_activation(
                x, weight, static_cast<const half*>(x_fp16), make_output(Fp8GdnInputShardGeometry{}),
                weight.n, stream);
        } else {
            launch_fp8_volta_qpn_with_fp16_activation(
                x, weight, static_cast<const half*>(x_fp16), make_output(Fp8GdnInputGeometry{}),
                weight.n, stream);
        }
    } else {
        if (weight.n == Fp8GdnInputShardGeometry::kOutputRows) {
            launch_fp8_volta_qpn_with_output(x, weight, make_output(Fp8GdnInputShardGeometry{}),
                                             weight.n, stream);
        } else {
            launch_fp8_volta_qpn_with_output(x, weight, make_output(Fp8GdnInputGeometry{}),
                                             weight.n, stream);
        }
    }
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
