#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_small_t.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
        Geometry::kOutputRows,
    };
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
struct LauncherTable {
    static constexpr auto kTable =
        make_launchers<Geometry>(std::make_index_sequence<kFp8LinearSmallTMax<Geometry> -
                                                          kFp8FirstSmallT + 1>{});
};

} // namespace

void fp8_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    if (weight.n == Fp8AttnInputShardGeometry::kOutputRows) {
        if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LinearSmallTMax<Fp8AttnInputShardGeometry>) {
            throw std::invalid_argument("fp8 attn_input_proj small-T: unsupported T");
        }
        constexpr auto& table = LauncherTable<Fp8AttnInputShardGeometry>::kTable;
        table[static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT)](x, weight, q, gate, k, v,
                                                                   stream);
        return;
    }
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LinearSmallTMax<Fp8AttnInputGeometry>) {
        throw std::invalid_argument("fp8 attn_input_proj small-T: unsupported T");
    }
    constexpr auto& table = LauncherTable<Fp8AttnInputGeometry>::kTable;
    table[static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT)](x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
