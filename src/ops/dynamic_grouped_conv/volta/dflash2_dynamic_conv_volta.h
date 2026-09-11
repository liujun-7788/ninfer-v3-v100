#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// sm_70 DFlash2 dynamic grouped convolution.
// prepare: `delta` is the full W_delta projection [side(2) x tap(2) x group(320), col].
// add:     `finish_delta` is the side-1 gains stashed by prepare, [group(320) x tap(2) x col].
void dflash2_bf16_control_projection_launch(const __nv_bfloat16* w, const __nv_bfloat16* x,
                                           int n_rows, int k, int cols, __nv_bfloat16* out,
                                           cudaStream_t stream);

void dflash2_dynamic_conv_prepare_launch(const __nv_bfloat16* n, const __nv_bfloat16* delta,
                                         const __nv_bfloat16* base, int width, int batch,
                                         __nv_bfloat16* prepared, __nv_bfloat16* finish_delta,
                                         cudaStream_t stream);

void dflash2_dynamic_conv_finish_add_launch(const __nv_bfloat16* projected,
                                            const __nv_bfloat16* finish_delta,
                                            const __nv_bfloat16* base, int width, int batch,
                                            __nv_bfloat16* residual, cudaStream_t stream);

} // namespace ninfer::ops::detail
