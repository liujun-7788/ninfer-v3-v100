#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Fused row order q|k|v|z with qkv taking 5/8 of the parent rows and z 3/8; holds for the full
// projection (16384 = 10240|6144) and the TP2 head-split shard (8192 = 5120|3072) alike.
template <class GeometryT>
struct Fp8GdnInputOutput {
    static constexpr std::int32_t kQkvRows = GeometryT::kOutputRows * 5 / 8;
    static constexpr std::int32_t kZRows   = GeometryT::kOutputRows * 3 / 8;

    __nv_bfloat16* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kQkvRows) {
            return qkv + static_cast<std::int64_t>(token) * kQkvRows + parent_row;
        }
        return z + static_cast<std::int64_t>(token) * kZRows + parent_row - kQkvRows;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        *destination(parent_row, token) = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

} // namespace ninfer::ops::detail
