#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// sm_70 coherent-path selector. Greedy (temperature <= 0) picks the lowest rank attaining
// max(edge) with a one-hot q. Positive-temperature rows draw from softmax(edge / temperature)
// against the counter-based RNG (seed, base_pos + i, kSamplePurposeDFlash2Proposal).
void dflash2_candidate_selector_walk_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                            const Tensor& projected_hidden, const Tensor& anchors,
                                            const Tensor& base_positions,
                                            const SamplingConfig* configs,
                                            const Tensor& predecessor_codebook,
                                            const Tensor& successor_codebook, Tensor& drafts,
                                            Tensor& proposal_q, cudaStream_t stream);

} // namespace ninfer::ops::detail
