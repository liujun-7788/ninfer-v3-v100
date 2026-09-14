#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/dense/packed/kernel.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kPackedAttentionVoltaThreads = 128;
inline constexpr int kPackedAttentionVoltaQueriesPerBlock =
    kPackedAttentionVoltaThreads / kWarpSize;

__launch_bounds__(kPackedAttentionVoltaThreads, 1) __global__ void
packed_attention_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ cu_seqlens,
    int segments, int uniform_segment_length, int tokens, __nv_bfloat16* __restrict__ out,
    std::int64_t q_stride_d, std::int64_t q_stride_h, std::int64_t q_stride_t,
    std::int64_t k_stride_d, std::int64_t k_stride_h, std::int64_t k_stride_t,
    std::int64_t v_stride_d, std::int64_t v_stride_h, std::int64_t v_stride_t) {
    constexpr int D = kPackedAttentionHeadDim;
    constexpr float Scale = 0.11785113019775792073f;

    const int lane  = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp  = static_cast<int>(threadIdx.x) / kWarpSize;
    const int token = static_cast<int>(blockIdx.x) * kPackedAttentionVoltaQueriesPerBlock + warp;
    const int head  = static_cast<int>(blockIdx.y);
    if (token >= tokens) { return; }

    int begin = 0;
    int end   = 0;
    if (lane == 0) {
        if (uniform_segment_length > 0) {
            begin = (token / uniform_segment_length) * uniform_segment_length;
            end   = min(tokens, begin + uniform_segment_length);
        } else {
            for (int segment = 0; segment < segments; ++segment) {
                const int candidate_begin = cu_seqlens[segment];
                const int candidate_end   = cu_seqlens[segment + 1];
                if (token >= candidate_begin && token < candidate_end) {
                    begin = candidate_begin;
                    end   = candidate_end;
                    break;
                }
            }
        }
    }
    begin = __shfl_sync(kFullWarpMask, begin, 0);
    end   = __shfl_sync(kFullWarpMask, end, 0);

    float q_values[3] = {};
    float acc[3]      = {};
#pragma unroll
    for (int item = 0; item < 3; ++item) {
        const int d = lane + item * kWarpSize;
        if (d < D) {
            q_values[item] = __bfloat162float(
                *packed_attention_ptr(q, q_stride_d, q_stride_h, q_stride_t, d, head, token));
        }
    }
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    for (int key_token = begin; key_token < end; ++key_token) {
        float dot = 0.0f;
#pragma unroll
        for (int item = 0; item < 3; ++item) {
            const int d = lane + item * kWarpSize;
            if (d < D) {
                dot += q_values[item] * __bfloat162float(*packed_attention_ptr(
                                            k, k_stride_d, k_stride_h, k_stride_t, d, head,
                                            key_token));
            }
        }
        dot = warp_reduce_sum(dot);

        float old_scale   = 0.0f;
        float probability = 0.0f;
        if (lane == 0) {
            const float score    = dot * Scale;
            const float next_max = fmaxf(row_max, score);
            old_scale = row_max == -CUDART_INF_F
                            ? 0.0f
                            : exp2_approx((row_max - next_max) * 1.4426950408889634f);
            probability = exp2_approx((score - next_max) * 1.4426950408889634f);
            row_sum      = row_sum * old_scale + probability;
            row_max      = next_max;
        }
        old_scale   = __shfl_sync(kFullWarpMask, old_scale, 0);
        probability = __shfl_sync(kFullWarpMask, probability, 0);
#pragma unroll
        for (int item = 0; item < 3; ++item) {
            const int d = lane + item * kWarpSize;
            if (d < D) {
                const float value = __bfloat162float(*packed_attention_ptr(
                    v, v_stride_d, v_stride_h, v_stride_t, d, head, key_token));
                acc[item] = acc[item] * old_scale + probability * value;
            }
        }
    }

    row_sum = __shfl_sync(kFullWarpMask, row_sum, 0);
#pragma unroll
    for (int item = 0; item < 3; ++item) {
        const int d = lane + item * kWarpSize;
        if (d < D) {
            out[(static_cast<std::int64_t>(token) * kPackedAttentionHeads + head) * D + d] =
                __float2bfloat16_rn(row_sum > 0.0f ? acc[item] / row_sum : 0.0f);
        }
    }
}

} // namespace ninfer::ops
