// DFlash2 port — sm_70 kernel stubs.
//
// Upstream's DFlash2 op kernels (dynamic grouped conv, candidate-selector path, context KV
// materialize, feature/top-k projections, rmsnorm+rope, rmsnorm pack-tail, the bf16 n256 k5120
// selector-hidden projection) are sm_80+ only (cp.async, bf16 mma, __reduce_*_sync). Their
// translation units are excluded from the Volta build (see src/CMakeLists.txt) until the port
// lands real Volta kernels (plan P2–P4). These stubs satisfy the link so the base model and
// the DFlash2 weight bindings (ValidateOnly) build; every entry throws if actually reached,
// which only happens under `--spec dflash2`.

#include "ops/context_kv_materialize/launch.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_kernels.h"
#include "ops/linear/w8/w8_feature.h"
#include "ops/rmsnorm_rope/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

namespace {
[[noreturn]] void unported(const char* op) {
    throw std::logic_error(std::string("DFlash2 op '") + op +
                           "' has no sm_70 kernel yet (V100 port P2+)");
}
} // namespace

void bf16_dynamic_grouped_conv_prepare_partial_launch(DynamicConvPrepareRoute, const Tensor&,
                                                      const Weight&, float*, cudaStream_t) {
    unported("dynamic_grouped_conv_prepare_partial");
}
void bf16_dynamic_grouped_conv_prepare_reduce_launch(DynamicConvPrepareRoute, const Tensor&,
                                                     const float*, Tensor&, Tensor&, cudaStream_t) {
    unported("dynamic_grouped_conv_prepare_reduce");
}
void w8_dynamic_grouped_conv_add_materialized_launch(W8DynamicConvAddSchedule, const Tensor&,
                                                     const Weight&, const Tensor&, const Tensor&,
                                                     Tensor&, Tensor&, cudaStream_t) {
    unported("w8_dynamic_grouped_conv_add_materialized");
}
void context_kv_materialize_launch(
    const Tensor&, const Tensor&, const Tensor&, const Tensor&,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>&,
    ContextKVMaterializeExecutionEnvelope, ContextKVMaterializeRoute, const Tensor&, cudaStream_t) {
    unported("context_kv_materialize");
}
void launch_w8_feature_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    unported("w8_feature_small_t");
}
void launch_w8_feature_r16_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    unported("w8_feature_r16_c64");
}
void launch_w8_feature_r32_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t) {
    unported("w8_feature_r32_c64");
}
void rmsnorm_rope_pair_launch(const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                              int, cudaStream_t) {
    unported("rmsnorm_rope_pair");
}
void rmsnorm_rope_single_launch(const Tensor&, const Tensor&, Tensor&, int, cudaStream_t) {
    unported("rmsnorm_rope_single");
}

} // namespace ninfer::ops::detail
