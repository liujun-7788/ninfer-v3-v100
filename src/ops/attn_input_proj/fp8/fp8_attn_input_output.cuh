#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Fused row order q | key | gate | value with q_rows = gate_rows = 3/14 of the parent rows and
// key_rows = value_rows = 1/14; holds for the full projection (14336) and the TP2 head-split
// shard (7168) alike. Boundaries are runtime values so both geometries share the kernels.
struct Fp8AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t key_begin;
    std::int32_t gate_begin;
    std::int32_t value_begin;

    __host__ __device__ Fp8AttentionInputOutput(__nv_bfloat16* query_ptr, __nv_bfloat16* key_ptr,
                                                __nv_bfloat16* gate_ptr, __nv_bfloat16* value_ptr,
                                                std::int32_t total_rows)
        : query(query_ptr),
          key(key_ptr),
          gate(gate_ptr),
          value(value_ptr),
          query_rows(total_rows * 3 / 7),
          key_rows(total_rows / 14),
          key_begin(query_rows),
          gate_begin(key_begin + key_rows),
          value_begin(gate_begin + query_rows) {}

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < key_begin) {
            return query + static_cast<std::int64_t>(token) * query_rows + parent_row;
        }
        if (parent_row < gate_begin) {
            return key + static_cast<std::int64_t>(token) * key_rows + parent_row - key_begin;
        }
        if (parent_row < value_begin) {
            return gate + static_cast<std::int64_t>(token) * query_rows + parent_row - gate_begin;
        }
        return value + static_cast<std::int64_t>(token) * key_rows + parent_row - value_begin;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        *destination(parent_row, token) = __float2bfloat16_rn(result);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

} // namespace ninfer::ops::detail
