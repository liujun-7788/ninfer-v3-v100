#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

// Fused row order: q | key | gate | value with q_rows = gate_rows = 3/14 of the parent rows
// and key_rows = value_rows = 1/14. Holds for the full projection (14336 = 6144|1024|6144|1024)
// and the TP2 head-split shard (7168 = 3072|512|3072|512) alike.
template <class Geometry>
struct Nvfp4AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    static constexpr std::int32_t kQueryRows  = Geometry::kOutputRows * 3 / 7;
    static constexpr std::int32_t kKeyRows    = Geometry::kOutputRows / 14;
    static constexpr std::int32_t kKeyBegin   = kQueryRows;
    static constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
    static constexpr std::int32_t kValueBegin = kGateBegin + kQueryRows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t,
                                          float result) const {
        const __nv_bfloat16 result_bf16 = __float2bfloat16_rn(result);
        if (parent_row < kKeyBegin) {
            query[parent_row] = result_bf16;
        } else if (parent_row < kGateBegin) {
            key[parent_row - kKeyBegin] = result_bf16;
        } else if (parent_row < kValueBegin) {
            gate[parent_row - kGateBegin] = result_bf16;
        } else {
            value[parent_row - kValueBegin] = result_bf16;
        }
    }
};

template <class Geometry>
void launch_decode(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                   Tensor& v, cudaStream_t stream) {
    using Output = Nvfp4AttentionInputOutput<Geometry>;
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    static_assert((Output::kQueryRows % 128) == 0);
    static_assert((Output::kKeyRows % 128) == 0);

    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    if (weight.n == Nvfp4AttnInputGeometry::kOutputRows) {
        launch_decode<Nvfp4AttnInputGeometry>(x, weight, q, gate, k, v, stream);
        return;
    }
    if (weight.n == Nvfp4AttnInputShardGeometry::kOutputRows) {
        launch_decode<Nvfp4AttnInputShardGeometry>(x, weight, q, gate, k, v, stream);
        return;
    }
    throw std::invalid_argument("nvfp4 attn_input decode: unsupported weight rows");
}

} // namespace ninfer::ops::detail
