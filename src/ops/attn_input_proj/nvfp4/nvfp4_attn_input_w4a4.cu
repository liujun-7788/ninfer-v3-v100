#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Fused row order q | key | gate | value with q_rows = gate_rows = 3/14 of the parent rows and
// key_rows = value_rows = 1/14; holds for the full projection and the TP2 head-split shard.
template <class GeometryT>
struct Nvfp4W4a4AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    static constexpr std::int32_t kQueryRows  = GeometryT::kOutputRows * 3 / 7;
    static constexpr std::int32_t kKeyRows    = GeometryT::kOutputRows / 14;
    static constexpr std::int32_t kKeyBegin   = kQueryRows;
    static constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
    static constexpr std::int32_t kValueBegin = kGateBegin + kQueryRows;

    static_assert((kQueryRows % 128) == 0);
    static_assert((kKeyRows % 128) == 0);

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kQueryRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

using M32N64                      = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128                     = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128                     = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined           = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident            = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
constexpr std::int32_t kTmaBlockM = 256;

template <class GeometryT, class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4W4a4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(GeometryT::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4W4a4AttentionOutput<GeometryT> output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<GeometryT, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <class GeometryT>
void launch_gemm_ladder(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                        Nvfp4W4a4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<GeometryT, M32N64>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<GeometryT, M32N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<GeometryT, M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<GeometryT, M64N128>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<GeometryT, M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<GeometryT, M128N128Pipelined>(weight, q, gate, k, v, workspace, tokens, stream);
    } else {
        launch_gemm<GeometryT, M128N128Resident>(weight, q, gate, k, v, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    // The shard skips the TMA route: its launcher still targets the full 14336-row projection.
    if (weight.n == Nvfp4AttnInputShardGeometry::kOutputRows) {
        launch_gemm_ladder<Nvfp4AttnInputShardGeometry>(weight, q, gate, k, v, workspace, tokens,
                                                        stream);
        return;
    }
    if (weight.n == Nvfp4AttnInputGeometry::kOutputRows) {
        if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {
            const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
            launch_nvfp4_w4a4_tma_attention(
                workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(gate.data),
                static_cast<__nv_bfloat16*>(k.data), static_cast<__nv_bfloat16*>(v.data), tokens,
                alpha, stream);
            return;
        }
        launch_gemm_ladder<Nvfp4AttnInputGeometry>(weight, q, gate, k, v, workspace, tokens,
                                                   stream);
        return;
    }
    throw std::invalid_argument("nvfp4 attn_input w4a4: unsupported weight rows");
}

} // namespace ninfer::ops::detail
