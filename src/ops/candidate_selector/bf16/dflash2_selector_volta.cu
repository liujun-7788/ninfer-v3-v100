// DFlash2 coherent-path candidate selector — sm_70.
//
// For each request, walk positions 0..K-1 left to right. At position i the predecessor token is
// the anchor (i=0) or the token chosen at i-1. The rank-256 edge score for candidate c is
//
//   edge[c] = unary[c,i,b] + sum_r pred_cb[r, pred] * g[r,i,b] * succ_cb[r, cand_ids[c,i,b]]
//
// Greedy (temperature <= 0): pick the lowest rank attaining max(edge); q is its one-hot.
// Stochastic (temperature > 0): q is softmax(edge / temperature) over the 16 candidates, and the
// draw is the inverse-CDF of q against sampling_uniform(seed, base_pos + i, DFlash2Proposal, 0)
// -- the same counter-based RNG contract as upstream, safe under CUDA-graph replay.

#include "ops/candidate_selector/bf16/dflash2_selector_volta.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include "ninfer/ops/sampling.h"

#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kRank       = 256;
constexpr int kCandidates = 16;

__global__ void dflash2_selector_walk_kernel(
    const std::int32_t* __restrict__ cand_ids, const float* __restrict__ unary,
    const __nv_bfloat16* __restrict__ g, const std::int32_t* __restrict__ anchors,
    const std::int32_t* __restrict__ base_positions, const SamplingConfig* __restrict__ configs,
    const __nv_bfloat16* __restrict__ pred_cb, const __nv_bfloat16* __restrict__ succ_cb, int steps,
    std::int32_t* __restrict__ drafts, float* __restrict__ proposal_q) {
    const int b = static_cast<int>(blockIdx.x);
    const int r = static_cast<int>(threadIdx.x); // 0..255

    __shared__ float sums[8];
    __shared__ float edge[kCandidates];
    __shared__ int predecessor;
    __shared__ float temperature;
    __shared__ unsigned long long seed;
    __shared__ int base_pos;
    if (r == 0) {
        predecessor = anchors[b];
        temperature = configs[b].temperature;
        seed        = configs[b].seed;
        base_pos    = base_positions[b];
    }
    __syncthreads();

    for (int i = 0; i < steps; ++i) {
        const int col        = i + steps * b;
        const int pred       = predecessor;
        const float pred_g_r = __bfloat162float(pred_cb[r + kRank * pred]) *
                               __bfloat162float(g[r + kRank * col]);

        for (int c = 0; c < kCandidates; ++c) {
            const int cid   = cand_ids[c + kCandidates * col];
            float contrib   = pred_g_r * __bfloat162float(succ_cb[r + kRank * cid]);
            contrib         = block_reduce_sum<256>(contrib, sums);
            if (r == 0) { edge[c] = unary[c + kCandidates * col] + contrib; }
            __syncthreads();
        }

        if (r == 0) {
            float best = edge[0];
            for (int c = 1; c < kCandidates; ++c) { best = fmaxf(best, edge[c]); }

            int j;
            if (temperature <= 0.0f) {
                j          = 0;
                float top  = edge[0];
                for (int c = 1; c < kCandidates; ++c) {
                    if (edge[c] > top) {
                        top = edge[c];
                        j   = c;
                    }
                }
                for (int c = 0; c < kCandidates; ++c) {
                    proposal_q[c + kCandidates * col] = (c == j) ? 1.0f : 0.0f;
                }
            } else {
                float wsum = 0.0f;
                float w[kCandidates];
                for (int c = 0; c < kCandidates; ++c) {
                    w[c] = __expf((edge[c] - best) / temperature);
                    wsum += w[c];
                }
                for (int c = 0; c < kCandidates; ++c) {
                    proposal_q[c + kCandidates * col] = w[c] / wsum;
                }
                const float u = sampling_uniform(seed, base_pos + i,
                                                 kSamplePurposeDFlash2Proposal, 0U);
                j              = kCandidates - 1;
                float cdf      = 0.0f;
                for (int c = 0; c < kCandidates; ++c) {
                    cdf += w[c] / wsum;
                    if (u < cdf) {
                        j = c;
                        break;
                    }
                }
            }

            const int chosen = cand_ids[j + kCandidates * col];
            drafts[col]      = chosen;
            predecessor      = chosen;
        }
        __syncthreads();
    }
}

} // namespace

void dflash2_candidate_selector_walk_launch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                            const Tensor& projected_hidden, const Tensor& anchors,
                                            const Tensor& base_positions,
                                            const SamplingConfig* configs,
                                            const Tensor& predecessor_codebook,
                                            const Tensor& successor_codebook, Tensor& drafts,
                                            Tensor& proposal_q, cudaStream_t stream) {
    const int steps = candidate_ids.ne[1];
    const int batch = candidate_ids.ne[2];
    dflash2_selector_walk_kernel<<<batch, kRank, 0, stream>>>(
        static_cast<const std::int32_t*>(candidate_ids.data),
        static_cast<const float*>(unary_scores.data),
        static_cast<const __nv_bfloat16*>(projected_hidden.data),
        static_cast<const std::int32_t*>(anchors.data),
        static_cast<const std::int32_t*>(base_positions.data), configs,
        static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
        static_cast<const __nv_bfloat16*>(successor_codebook.data), steps,
        static_cast<std::int32_t*>(drafts.data), static_cast<float*>(proposal_q.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
