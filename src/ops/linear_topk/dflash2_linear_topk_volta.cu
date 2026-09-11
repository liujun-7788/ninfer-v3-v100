// DFlash2 unary candidate top-16 — sm_70.
//
// Per column of the proposal-head logits, return the 16 highest scores and their global token ids,
// ordered by descending score with exact ties broken by lower id. Physical rows >= valid_rows are
// excluded (full head); the optimized head remaps every row through id_map.

#include "ops/linear_topk/dflash2_linear_topk_volta.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>

namespace ninfer::ops::detail {
namespace {

constexpr int kTopK   = 16;
constexpr int kBlock  = 256;
constexpr int kWarps  = kBlock / 32;

struct Cand {
    float score;
    int   id;
};

// a strictly beats b under (score desc, id asc)
__device__ __forceinline__ bool better(float sa, int ia, float sb, int ib) {
    return sa > sb || (sa == sb && ia < ib);
}

__device__ __forceinline__ void local_insert(float* s, int* d, float score, int id) {
    if (!better(score, id, s[kTopK - 1], d[kTopK - 1])) { return; }
    int p = kTopK - 1;
    while (p > 0 && better(score, id, s[p - 1], d[p - 1])) {
        s[p] = s[p - 1];
        d[p] = d[p - 1];
        --p;
    }
    s[p] = score;
    d[p] = id;
}

__global__ void dflash2_topk16_kernel(const __nv_bfloat16* __restrict__ logits,
                                      const std::int32_t* __restrict__ id_map, int head_rows,
                                      int valid_rows, int columns, std::int32_t* __restrict__ ids,
                                      float* __restrict__ scores) {
    const int col = static_cast<int>(blockIdx.x);
    if (col >= columns) { return; }
    const __nv_bfloat16* logit_col = logits + static_cast<std::int64_t>(col) * head_rows;

    float ls[kTopK];
    int   ld[kTopK];
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        ls[i] = -FLT_MAX;
        ld[i] = 0x7fffffff;
    }

    for (int r = static_cast<int>(threadIdx.x); r < valid_rows; r += kBlock) {
        const int gid = id_map != nullptr ? id_map[r] : r;
        local_insert(ls, ld, __bfloat162float(logit_col[r]), gid);
    }

    __shared__ float sh_score[kBlock * kTopK];
    __shared__ int   sh_id[kBlock * kTopK];
#pragma unroll
    for (int i = 0; i < kTopK; ++i) {
        sh_score[threadIdx.x * kTopK + i] = ls[i];
        sh_id[threadIdx.x * kTopK + i]    = ld[i];
    }
    __syncthreads();

    // 16 selection rounds over the 4096 shared candidates. Single-threaded: at most
    // 16 * 4096 comparisons per column, negligible next to the projection.
    if (threadIdx.x == 0) {
        const int total = kBlock * kTopK;
        for (int round = 0; round < kTopK; ++round) {
            float best_s = -FLT_MAX;
            int   best_e = -1;
            for (int e = 0; e < total; ++e) {
                if (best_e < 0 || better(sh_score[e], sh_id[e], best_s, sh_id[best_e])) {
                    best_s = sh_score[e];
                    best_e = e;
                }
            }
            scores[static_cast<std::int64_t>(col) * kTopK + round] = best_s;
            ids[static_cast<std::int64_t>(col) * kTopK + round]    = best_e >= 0 ? sh_id[best_e] : 0;
            if (best_e >= 0) { sh_score[best_e] = -FLT_MAX; }
        }
    }
}

} // namespace

void dflash2_linear_topk16_launch(const Tensor& logits, const std::int32_t* id_map,
                                  std::int32_t valid_rows, Tensor& ids, Tensor& scores,
                                  cudaStream_t stream) {
    const int head_rows = logits.ne[0];
    const int columns   = logits.ne[1];
    dflash2_topk16_kernel<<<columns, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), id_map, head_rows, valid_rows, columns,
        static_cast<std::int32_t*>(ids.data), static_cast<float*>(scores.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
