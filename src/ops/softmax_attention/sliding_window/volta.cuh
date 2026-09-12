#pragma once

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/softmax_attention/common/context_query.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kSlidingWindowVoltaThreads = kContextQueryHeadDim;

// DFlash2's production K=7 round is an eight-column masked block.  The original sm_70
// fallback launched one CTA per (query token, query head), so all eight CTAs streamed the
// same 2K K/V window independently.  This split-KV path makes the proposal width the unit
// of work: one warp owns one query token, while the eight warps in a CTA share each K/V
// vector loaded from the cyclic cache.  Splitting the window supplies enough CTAs for an
// 80-SM V100 and preserves the existing partial/reduce contract.
inline constexpr int kSlidingWindowVoltaOctetThreads = 8 * 32;

template <int Tokens>
__launch_bounds__(kSlidingWindowVoltaOctetThreads, 2) __global__ void
sliding_window_attention_volta_octet_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k, const __half* __restrict__ context_v,
    int padded_context, int max_context, int split_capacity, float scale,
    float* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l) {
    static_assert(Tokens == 8, "the Volta octet path is the DFlash2 K=7 tile");
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int D                 = kContextQueryHeadDim;
    constexpr int Window            = 2048;
    constexpr int KeyBlock          = 32;
    constexpr std::int64_t QElems   = static_cast<std::int64_t>(D) * kContextQueryQHeads * Tokens;
    constexpr std::int64_t KVElems  = static_cast<std::int64_t>(D) * kContextQueryKVHeads * Tokens;
    constexpr std::int64_t StatElems = static_cast<std::int64_t>(kContextQueryQHeads) * Tokens;
    constexpr float Log2E           = 1.4426950408889634074f;
    constexpr unsigned FullMask     = 0xffffffffu;

    __shared__ float key_s[D];
    __shared__ float value_s[D];

    const int q_head = static_cast<int>(blockIdx.x);
    const int split  = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int token  = warp;
    const int valid  = valid_columns[batch];
    const int length = positions[static_cast<std::int64_t>(batch) * Tokens];

    if (q_head >= kContextQueryQHeads || split >= split_capacity || valid < 0 || valid > Tokens ||
        length < 0 || length > max_context) {
        return;
    }

    q += QElems * batch;
    query_k += KVElems * batch;
    query_v += KVElems * batch;
    positions += static_cast<std::int64_t>(Tokens) * batch;
    partial_acc += QElems * split_capacity * batch;
    partial_m += StatElems * split_capacity * batch;
    partial_l += StatElems * split_capacity * batch;

    const int context_count = min(length, Window);
    const int context_start = length - context_count;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    if (split >= active_splits) { return; }

    const int tile_begin =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * split) / active_splits);
    const int tile_end = static_cast<int>(
        (static_cast<std::int64_t>(context_tiles) * (split + 1)) / active_splits);
    const int key_begin = context_start + tile_begin * KeyBlock;
    const int key_end   = min(length, context_start + tile_end * KeyBlock);
    const bool owns_query = split == active_splits - 1;
    const int kv_head     = q_head / kContextQueryGroup;
    const bool live       = token < valid;
    const int query_position = live ? positions[token] : 0;

    float q_frag[4];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d = lane + item * 32;
        q_frag[item] = live ? __bfloat162float(q[context_query_q_index(q_head, d, token)]) : 0.0f;
    }
    float row_m = -CUDART_INF_F;
    float row_l = 0.0f;

    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[batch]) * D * padded_context * kContextQueryKVHeads;
    for (int key = key_begin; key < key_end; ++key) {
        if (tid < D) {
            const int slot = key & (Window - 1);
            const std::int64_t index = lane_base + tid + static_cast<std::int64_t>(D) *
                                                               (slot + padded_context * kv_head);
            key_s[tid]   = __bfloat162float(context_k[index]);
            value_s[tid] = __half2float(context_v[index]);
        }
        __syncthreads();

        float dot = 0.0f;
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * 32;
            dot += q_frag[item] * key_s[d];
        }
        dot = warp_sum<32>(dot, FullMask);
        dot = __shfl_sync(FullMask, dot, 0);
        const bool allowed = live && abs(key - query_position) < Window;
        const float score  = allowed ? dot * scale : -CUDART_INF_F;
        const float next_m = fmaxf(row_m, score);
        const float alpha  = row_m == -CUDART_INF_F ? 0.0f
                                                     : exp2_approx((row_m - next_m) * Log2E);
        const float probability = score == -CUDART_INF_F
                                      ? 0.0f
                                      : exp2_approx((score - next_m) * Log2E);
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * 32;
            acc[item] = acc[item] * alpha + probability * value_s[d];
        }
        row_l = row_l * alpha + probability;
        row_m = next_m;
        __syncthreads();
    }

    if (owns_query) {
        for (int key_token = 0; key_token < valid; ++key_token) {
            if (tid < D) {
                const auto index = context_query_query_kv_index(kv_head, tid, key_token);
                key_s[tid]       = __bfloat162float(query_k[index]);
                value_s[tid]     = __bfloat162float(query_v[index]);
            }
            __syncthreads();

            float dot = 0.0f;
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int d = lane + item * 32;
                dot += q_frag[item] * key_s[d];
            }
            dot = warp_sum<32>(dot, FullMask);
            dot = __shfl_sync(FullMask, dot, 0);
            const bool allowed =
                live && abs(positions[key_token] - query_position) < Window;
            const float score  = allowed ? dot * scale : -CUDART_INF_F;
            const float next_m = fmaxf(row_m, score);
            const float alpha  = row_m == -CUDART_INF_F ? 0.0f
                                                         : exp2_approx((row_m - next_m) * Log2E);
            const float probability = score == -CUDART_INF_F
                                          ? 0.0f
                                          : exp2_approx((score - next_m) * Log2E);
#pragma unroll
            for (int item = 0; item < 4; ++item) {
                const int d = lane + item * 32;
                acc[item] = acc[item] * alpha + probability * value_s[d];
            }
            row_l = row_l * alpha + probability;
            row_m = next_m;
            __syncthreads();
        }
    }

    if (live) {
        if (lane == 0) {
            const auto stat = context_query_stat_index<Tokens>(q_head, token, split);
            partial_m[stat] = row_m;
            partial_l[stat] = row_l;
        }
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d = lane + item * 32;
            partial_acc[context_query_partial_index<Tokens>(q_head, d, token, split)] = acc[item];
        }
    }
#endif
}

__launch_bounds__(kSlidingWindowVoltaThreads, 1) __global__ void
sliding_window_attention_volta_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k,
    const __half* __restrict__ context_v, int padded_context, int tokens, int window,
    float scale, __nv_bfloat16* __restrict__ out) {
    constexpr int D       = kContextQueryHeadDim;
    constexpr int QHeads  = kContextQueryQHeads;
    constexpr int KVHeads = kContextQueryKVHeads;
    // Window is 2048 (DFlash2 local layers) or 4096 (DFlash1); both are powers of two so the
    // ring slot stays `key & (Window - 1)`.
    const int Window = window;

    __shared__ float warp_sums[kSlidingWindowVoltaThreads / kWarpSize];
    __shared__ float score_s;

    const int token  = static_cast<int>(blockIdx.x);
    const int q_head = static_cast<int>(blockIdx.y);
    const int batch  = static_cast<int>(blockIdx.z);
    const int d      = static_cast<int>(threadIdx.x);
    const int valid  = valid_columns[batch];
    const std::int64_t q_batch = static_cast<std::int64_t>(batch) * D * QHeads * tokens;
    const std::int64_t kv_batch = static_cast<std::int64_t>(batch) * D * KVHeads * tokens;
    const std::int64_t out_index =
        q_batch + d + static_cast<std::int64_t>(D) * (q_head + QHeads * token);
    if (token >= valid) {
        out[out_index] = __float2bfloat16(0.0f);
        return;
    }

    const auto* batch_positions = positions + static_cast<std::int64_t>(batch) * tokens;
    const int context_end        = batch_positions[0];
    const int query_position     = batch_positions[token];
    const int context_begin      = max(0, max(context_end - Window, query_position - (Window - 1)));
    const int kv_head            = q_head / (QHeads / KVHeads);
    const float q_value          = __bfloat162float(q[out_index]);

    const std::int64_t lane_base =
        static_cast<std::int64_t>(lanes[batch]) * D * padded_context * KVHeads;
    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    float acc     = 0.0f;

    for (int key = context_begin; key < context_end; ++key) {
        const int slot = key & (Window - 1);
        const std::int64_t index = lane_base + d + static_cast<std::int64_t>(D) *
                                                       (slot + padded_context * kv_head);
        const float key_value = __bfloat162float(context_k[index]);
        const float value     = __half2float(context_v[index]);
        const float dot = block_reduce_sum<kSlidingWindowVoltaThreads>(q_value * key_value,
                                                                       warp_sums);
        if (d == 0) { score_s = dot * scale; }
        __syncthreads();
        const float score      = score_s;
        const float next_max   = fmaxf(row_max, score);
        const float old_scale  = row_max == -CUDART_INF_F
                                     ? 0.0f
                                     : exp2_approx((row_max - next_max) * 1.4426950408889634f);
        const float probability = exp2_approx((score - next_max) * 1.4426950408889634f);
        acc     = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
        __syncthreads();
    }

    for (int key_token = 0; key_token < valid; ++key_token) {
        const int key_position = batch_positions[key_token];
        if (abs(key_position - query_position) >= Window) { continue; }
        const std::int64_t index =
            kv_batch + d + static_cast<std::int64_t>(D) * (kv_head + KVHeads * key_token);
        const float key_value = __bfloat162float(query_k[index]);
        const float value     = __bfloat162float(query_v[index]);
        const float dot = block_reduce_sum<kSlidingWindowVoltaThreads>(q_value * key_value,
                                                                       warp_sums);
        if (d == 0) { score_s = dot * scale; }
        __syncthreads();
        const float score       = score_s;
        const float next_max    = fmaxf(row_max, score);
        const float old_scale   = row_max == -CUDART_INF_F
                                      ? 0.0f
                                      : exp2_approx((row_max - next_max) * 1.4426950408889634f);
        const float probability = exp2_approx((score - next_max) * 1.4426950408889634f);
        acc     = acc * old_scale + probability * value;
        row_sum = row_sum * old_scale + probability;
        row_max = next_max;
        __syncthreads();
    }

    out[out_index] = __float2bfloat16_rn(row_sum > 0.0f ? acc / row_sum : 0.0f);
}

} // namespace ninfer::ops
