#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& qkv, Tensor& z, Fp8A8Workspace workspace,
                std::int32_t tokens, cudaStream_t stream) {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    static_assert(((Geometry::kOutputRows * 5 / 8) % Schedule::kBlockRows) == 0 &&
                  ((Geometry::kOutputRows * 3 / 8) % Schedule::kBlockRows) == 0);
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Fp8GdnInputOutput<Geometry> output{static_cast<__nv_bfloat16*>(qkv.data),
                                             static_cast<__nv_bfloat16*>(z.data)};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Schedule, FullTokens, Fp8IdentityEpilogue,
                           Fp8GdnInputOutput<Geometry>>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_for(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                Fp8A8Workspace workspace, cudaStream_t stream) {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<Geometry, true>(weight, qkv, z, workspace, x.ne[1], stream);
    } else {
        launch_mma<Geometry, false>(weight, qkv, z, workspace, x.ne[1], stream);
    }
}

} // namespace

void fp8_gdn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                             Fp8A8Workspace workspace, cudaStream_t stream) {
    if (weight.n == Fp8GdnInputShardGeometry::kOutputRows) {
        launch_for<Fp8GdnInputShardGeometry>(x, weight, qkv, z, workspace, stream);
        return;
    }
    launch_for<Fp8GdnInputGeometry>(x, weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail
