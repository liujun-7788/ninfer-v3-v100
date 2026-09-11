#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace ninfer::ops::detail {

// sm_70 unary top-16: id_map == nullptr for the full head (ids are rows), otherwise the
// optimized head remaps each row. valid_rows bounds the participating physical rows.
void dflash2_linear_topk16_launch(const Tensor& logits, const std::int32_t* id_map,
                                  std::int32_t valid_rows, Tensor& ids, Tensor& scores,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
