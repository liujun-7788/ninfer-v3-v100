#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"

#include "ops/linear/nvfp4/nvfp4_volta_as_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

void nvfp4_volta_as_gemm_launch(const Tensor& x, const Weight& w, Tensor& out,
                                cudaStream_t stream) {
    using S = Nvfp4VoltaAsSchedule;
    const std::int32_t n = w.n;
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const float inverse_weight_divisor = 1.0F / w.weight_scale_divisor;
    const int qpn = w.layout == QuantLayout::VoltaQpnPrepacked ? 1 : 0;

    const dim3 grid(static_cast<unsigned>((n + S::kRowsPerCta - 1) / S::kRowsPerCta), 1,
                    static_cast<unsigned>((t + S::kTTile - 1) / S::kTTile));
    nvfp4_volta_as_kernel<<<grid, S::kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.scales),
        static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), out_ld,
        n, k, t, inverse_weight_divisor, qpn);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
