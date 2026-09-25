#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_launch.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/attn_input_proj/fp8/fp8_attn_input_cutlass_sm70.h"
#endif

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kVoltaCutlassMinT = 33;

enum class Fp8AttnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8AttnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 attn_input_proj: unsupported policy");
    }
    return tokens >= 11 ? Fp8AttnInputRoute::A8 : Fp8AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, WorkspaceArena* workspace, cudaStream_t stream) {
    // Fused row order q | key | gate | value: q/gate take 3/14 of the parent rows each and
    // k/v take 1/14 each, for both the full projection (14336) and the head-split shard (7168).
    const std::int32_t kQRows  = weight.n * 3 / 7;
    const std::int32_t kKvRows = weight.n / 14;
#ifdef NINFER_VOLTA_BUILD
    if (x.ne[1] >= kVoltaCutlassMinT) {
        if (workspace == nullptr) {
            throw std::invalid_argument("fp8 Volta attention prefill requires caller workspace");
        }
        fp8_attn_input_cutlass_sm70_launch(x, weight, q, gate, k, v, *workspace, stream);
        return;
    }
    const bool qpn = fp8_volta_qpn_supported(weight.n, weight.k, kFp8VoltaQpnMaxTokens);
    const std::int32_t kChunk =
        qpn ? kFp8VoltaQpnMaxTokens : kFp8LinearSmallTMax<Fp8AttnInputGeometry>;
    if (!qpn) { throw std::logic_error("fp8 Volta attention problem has no QPN route"); }
    std::optional<WorkspaceArena::Scope> scope;
    DeviceSpan activation;
    if (workspace != nullptr) {
        scope.emplace(workspace->scope());
        activation = workspace->alloc_bytes(
            static_cast<std::size_t>(weight.k) * std::min(x.ne[1], kChunk) * sizeof(std::uint16_t),
            256);
    }
#else
    constexpr std::int32_t kChunk  = kFp8LinearSmallTMax<Fp8AttnInputGeometry>;
#endif
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* query = static_cast<std::uint8_t*>(q.data) +
                      static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* output_gate = static_cast<std::uint8_t*>(gate.data) +
                            static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* key = static_cast<std::uint8_t*>(k.data) +
                    static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        auto* value = static_cast<std::uint8_t*>(v.data) +
                      static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor query_chunk(query, DType::BF16, {kQRows, active});
        Tensor gate_chunk(output_gate, DType::BF16, {kQRows, active});
        Tensor key_chunk(key, DType::BF16, {kKvRows, active});
        Tensor value_chunk(value, DType::BF16, {kKvRows, active});
#ifdef NINFER_VOLTA_BUILD
        if (fp8_volta_qpn_supported(weight.n, weight.k, active)) {
            if (activation.data != nullptr) {
                fp8_stage_bf16_activation_sm70(input_chunk, activation.data, stream);
            }
            launch_fp8_attn_input_volta_qpn(input_chunk, weight, query_chunk, activation.data,
                                            gate_chunk, key_chunk, value_chunk, stream);
            continue;
        }
#endif
        if (active == 1) {
            fp8_attn_input_decode_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                         value_chunk, stream);
        } else {
            fp8_attn_input_small_t_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                          value_chunk, stream);
        }
    }
}

} // namespace

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    std::size_t capacity = resolve_route(policy, max_tokens) == Fp8AttnInputRoute::A8
                               ? fp8_a8_workspace_capacity_bytes(
                                     max_tokens, Fp8AttnInputGeometry::kInputRows)
                               : 0;
#ifdef NINFER_VOLTA_BUILD
    if (max_tokens >= kVoltaCutlassMinT) {
        capacity = std::max(capacity, fp8_attn_input_cutlass_workspace_bytes(max_tokens));
    }
    capacity = std::max(capacity, static_cast<std::size_t>(Fp8AttnInputGeometry::kInputRows) *
                                      std::min(max_tokens, kFp8VoltaQpnMaxTokens) *
                                      sizeof(std::uint16_t));
#endif
    return capacity;
}

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 attn_input_proj requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
    fp8_attn_input_a8_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
