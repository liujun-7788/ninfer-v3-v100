#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <array>
#include <cstddef>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch       = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&, const Tensor&,
                        const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);
using RecordLaunch = void (*)(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens, class Publish>
void launch_exact(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                  const Tensor& conv_states, const Tensor& valid_columns,
                  const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                  Publish publish, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert(Schedule::kTokenTile == ActiveTokens);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    const GdnRowGeometry rows = GdnRowGeometry::from_parent_rows(weight.n);
    nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule, Nvfp4IdentityEpilogue,
                         GdnConvOutput<ActiveTokens, Publish>, Nvfp4SmallTFinalization::RowVector>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
            make_gdn_conv_output<ActiveTokens>(rows, conv_weight, conv_states, valid_columns,
                                               initial_slot, query, key, value, z, publish));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int ActiveTokens>
void launch_snapshot_exact(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                           Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                           Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                           cudaStream_t stream) {
    const GdnRowGeometry rows = GdnRowGeometry::from_parent_rows(weight.n);
    launch_exact<Geometry, ActiveTokens>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data),
                               rows.channels},
        stream);
}

template <class Geometry, int ActiveTokens>
void launch_record_exact(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                         const Tensor& conv_states, const Tensor& valid_columns,
                         const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                         Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    const GdnRowGeometry rows = GdnRowGeometry::from_parent_rows(weight.n);
    launch_exact<Geometry, ActiveTokens>(x, weight, conv_weight, conv_states, valid_columns,
                                         initial_slot, query, key, value, z,
                                         RecordColumnPublish{static_cast<__nv_bfloat16*>(
                                                                 conv_record.data),
                                                             rows.channels, ActiveTokens},
                                         stream);
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_snapshot_exact<Geometry, kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<RecordLaunch, sizeof...(Offsets)>{
        &launch_record_exact<Geometry, kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
struct LauncherTables {
    static constexpr auto kSnapshot =
        make_launchers<Geometry>(std::make_index_sequence<16 - kNvfp4FirstSmallT + 1>{});
    static constexpr auto kRecord =
        make_record_launchers<Geometry>(std::make_index_sequence<16 - kNvfp4FirstSmallT + 1>{});
};

} // namespace

void nvfp4_gdn_snapshot_small_t_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    if (weight.n == Nvfp4GdnInputShardGeometry::kOutputRows) {
        constexpr auto& table = LauncherTables<Nvfp4GdnInputShardGeometry>::kSnapshot;
        table[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                     snapshot_base_slot, query, key, value, z, stream);
        return;
    }
    constexpr auto& table = LauncherTables<Nvfp4GdnInputGeometry>::kSnapshot;
    table[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                 snapshot_base_slot, query, key, value, z, stream);
}

void nvfp4_gdn_record_small_t_launch(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, const Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_slot,
                                     Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                     Tensor& z, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    if (weight.n == Nvfp4GdnInputShardGeometry::kOutputRows) {
        constexpr auto& table = LauncherTables<Nvfp4GdnInputShardGeometry>::kRecord;
        table[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record,
                     query, key, value, z, stream);
        return;
    }
    constexpr auto& table = LauncherTables<Nvfp4GdnInputGeometry>::kRecord;
    table[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record,
                 query, key, value, z, stream);
}

} // namespace ninfer::ops::detail
