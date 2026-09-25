#pragma once

#include "targets/qwen3_6_27b/impl/load/bindings.h"

namespace ninfer::targets::qwen3_6_27b::detail {

// Compacts every tensor-parallel-shardable weight in the runtime view in place for the given
// rank (0 or 1). Each weight keeps its base payload pointer; only rows/columns, plane offsets,
// and the scale plane move, so the sharded payload always fits inside the original allocation.
// The caller must run this with the owning device current (the thread bound to the
// DeviceContext whose device holds the weights). All copies run on the default stream of the
// current device and the device is synchronized before return. Throws for split (non-fused)
// payloads or layouts the TP2 sharding code does not handle.
void tp_shard_model(RuntimeModelView& runtime, int rank);

} // namespace ninfer::targets::qwen3_6_27b::detail
