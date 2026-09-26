#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

namespace ninfer::ops::detail {
namespace {

using M32N64                      = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128                     = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128                     = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined           = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident            = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
constexpr std::int32_t kTmaBlockM = 256;

template <class GeometryT, class Schedule>
void launch_gemm(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(GeometryT::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<GeometryT, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        Nvfp4GdnInputOutput<GeometryT>{static_cast<__nv_bfloat16*>(qkv.data),
                                       static_cast<__nv_bfloat16*>(z.data)});
    CUDA_CHECK(cudaGetLastError());
}

template <class GeometryT>
void launch_gemm_ladder(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4W4a4Workspace workspace,
                        std::int32_t tokens, cudaStream_t stream) {
    if (tokens <= 64) {
        launch_gemm<GeometryT, M32N64>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<GeometryT, M32N128>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<GeometryT, M128N128Pipelined>(weight, qkv, z, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<GeometryT, M64N128>(weight, qkv, z, workspace, tokens, stream);
    } else {
        launch_gemm<GeometryT, M128N128Resident>(weight, qkv, z, workspace, tokens, stream);
    }
}

} // namespace

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    // The shard skips the TMA route: its launcher still targets the full 16384-row projection.
    if (weight.n == Nvfp4GdnInputShardGeometry::kOutputRows) {
        launch_gemm_ladder<Nvfp4GdnInputShardGeometry>(weight, qkv, z, workspace, tokens, stream);
        return;
    }
    if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_gdn(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(qkv.data),
            static_cast<__nv_bfloat16*>(z.data), tokens, alpha, stream);
        return;
    }
    launch_gemm_ladder<Nvfp4GdnInputGeometry>(weight, qkv, z, workspace, tokens, stream);
}

} // namespace ninfer::ops::detail
