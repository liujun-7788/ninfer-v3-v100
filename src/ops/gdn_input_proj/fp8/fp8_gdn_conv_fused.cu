#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_small_t.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using SnapshotLaunch = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&, const Tensor&,
                                const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                                cudaStream_t);
using RecordLaunch   = void (*)(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens, class Publish>
void launch_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                    const Tensor& conv_states, const Tensor& valid_columns,
                    const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                    Tensor& z, Publish publish, cudaStream_t stream) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert(Schedule::kTokenTile == ActiveTokens);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    using Output          = GdnConvOutput<ActiveTokens, Publish>;
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule, Output, Fp8IdentityEpilogue,
                       Fp8GemvIdentityRows, false, Fp8SmallTFinalization::RowVector>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            make_gdn_conv_output<ActiveTokens>(GdnRowGeometry::from_parent_rows(weight.n),
                                               conv_weight, conv_states, valid_columns,
                                               initial_slot, query, key, value, z, publish));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int ActiveTokens>
void launch_snapshot_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                             Tensor& conv_states, const Tensor& valid_columns,
                             const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                             Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                             cudaStream_t stream) {
    launch_small_t<Geometry, ActiveTokens>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data),
                               weight.n * 5 / 8},
        stream);
}

template <class Geometry, int ActiveTokens>
void launch_record_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                           const Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    launch_small_t<Geometry, ActiveTokens>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data), weight.n * 5 / 8,
                            ActiveTokens},
        stream);
}

template <class Geometry>
void launch_snapshot_decode(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                            Tensor& conv_states, const Tensor& valid_columns,
                            const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                            Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    using Schedule        = typename Fp8LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    fp8_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        make_gdn_conv_output<1>(
            GdnRowGeometry::from_parent_rows(weight.n), conv_weight, conv_states, valid_columns,
            initial_slot, query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   weight.n * 5 / 8}));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_snapshot_launchers(std::index_sequence<Offsets...>) {
    return std::array<SnapshotLaunch, sizeof...(Offsets)>{
        &launch_snapshot_small_t<Geometry, kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<RecordLaunch, sizeof...(Offsets)>{
        &launch_record_small_t<Geometry, kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
struct LauncherTables {
    static constexpr auto kSnapshot = make_snapshot_launchers<Geometry>(
        std::make_index_sequence<kFp8LinearSmallTMax<Geometry> - kFp8FirstSmallT + 1>{});
    static constexpr auto kRecord = make_record_launchers<Geometry>(
        std::make_index_sequence<kFp8LinearSmallTMax<Geometry> - kFp8FirstSmallT + 1>{});
};

} // namespace

void fp8_gdn_snapshot_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                   Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                   cudaStream_t stream) {
    if (x.ne[2] != 1 || x.ne[1] <= 0) {
        throw std::invalid_argument("fp8 GDN snapshot fused: unsupported B/W");
    }
    if (x.ne[1] == 1) {
        if (weight.n == Fp8GdnInputShardGeometry::kOutputRows) {
            launch_snapshot_decode<Fp8GdnInputShardGeometry>(x, weight, conv_weight, conv_states,
                                                             valid_columns, initial_slot,
                                                             snapshot_base_slot, query, key, value,
                                                             z, stream);
        } else {
            launch_snapshot_decode<Fp8GdnInputGeometry>(x, weight, conv_weight, conv_states,
                                                        valid_columns, initial_slot,
                                                        snapshot_base_slot, query, key, value, z,
                                                        stream);
        }
        return;
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT);
    if (weight.n == Fp8GdnInputShardGeometry::kOutputRows) {
        if (x.ne[1] > kFp8LinearSmallTMax<Fp8GdnInputShardGeometry>) {
            throw std::invalid_argument("fp8 GDN snapshot fused: unsupported B/W");
        }
        constexpr auto& table = LauncherTables<Fp8GdnInputShardGeometry>::kSnapshot;
        table[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                     snapshot_base_slot, query, key, value, z, stream);
        return;
    }
    if (x.ne[1] > kFp8LinearSmallTMax<Fp8GdnInputGeometry>) {
        throw std::invalid_argument("fp8 GDN snapshot fused: unsupported B/W");
    }
    constexpr auto& table = LauncherTables<Fp8GdnInputGeometry>::kSnapshot;
    table[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                 snapshot_base_slot, query, key, value, z, stream);
}

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    if (x.ne[2] != 1 || x.ne[1] < kFp8FirstSmallT) {
        throw std::invalid_argument("fp8 GDN record fused: unsupported B/W");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT);
    if (weight.n == Fp8GdnInputShardGeometry::kOutputRows) {
        if (x.ne[1] > kFp8LinearSmallTMax<Fp8GdnInputShardGeometry>) {
            throw std::invalid_argument("fp8 GDN record fused: unsupported B/W");
        }
        constexpr auto& table = LauncherTables<Fp8GdnInputShardGeometry>::kRecord;
        table[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record,
                     query, key, value, z, stream);
        return;
    }
    if (x.ne[1] > kFp8LinearSmallTMax<Fp8GdnInputGeometry>) {
        throw std::invalid_argument("fp8 GDN record fused: unsupported B/W");
    }
    constexpr auto& table = LauncherTables<Fp8GdnInputGeometry>::kRecord;
    table[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record,
                 query, key, value, z, stream);
}

} // namespace ninfer::ops::detail
