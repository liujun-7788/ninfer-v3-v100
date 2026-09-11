// DFlash2 candidate-selector hidden projection [256, 5120] — sm_70.
// Reuses the generic BF16 matvec (the fork's general BF16 linear does not admit this shape).

#include "ops/linear/bf16/bf16_launch.h"
#include "ops/dynamic_grouped_conv/volta/dflash2_dynamic_conv_volta.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void launch_bf16_n256_k5120(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    dflash2_bf16_control_projection_launch(static_cast<const __nv_bfloat16*>(weight.qdata),
                                           static_cast<const __nv_bfloat16*>(x.data),
                                           /*n_rows=*/256, /*k=*/x.ne[0], /*cols=*/x.ne[1],
                                           static_cast<__nv_bfloat16*>(out.data), stream);
}

} // namespace ninfer::ops::detail
