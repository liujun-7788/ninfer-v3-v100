#pragma once

#include "ops/softmax_attention/common/head_mapping.cuh"

namespace ninfer::ops {

#ifdef NINFER_VOLTA_BUILD
// The small-T policy requests roughly one split per 480 cached tokens. Keep that policy
// unconstrained through the supported 262,144-token Volta context; the former 85-split ceiling
// became binding near 41K and serialized progressively more KV work into each CTA.
inline constexpr int kSmallTMaximumBaseSplits = 560; // fork c58a92ba long-context split scaling; the small_t reduce folds over 256 threads so splits may exceed blockDim.x
#else
inline constexpr int kSmallTMaximumBaseSplits = 85;
#endif

template <int QHeadsValue, int KVHeadsValue, int SmallTSplitScaleValue>
struct CausalAttentionGeometry : AttentionHeadMapping<QHeadsValue, KVHeadsValue> {
    static_assert(SmallTSplitScaleValue > 0);

    static constexpr int SmallTSplitScale = SmallTSplitScaleValue;
    static constexpr int SmallTMaximumSplits =
        kSmallTMaximumBaseSplits * SmallTSplitScale;
};

using CausalD256H24Kv4 = CausalAttentionGeometry<24, 4, 1>;
using CausalD256H16Kv2 = CausalAttentionGeometry<16, 2, 2>;

} // namespace ninfer::ops
