#pragma once

#include "core/device.h"
#include "core/tp_group.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ninfer::tpexec {

// Process-local TP2 execution state. The group and the fixed per-rank/per-site
// scratch buffers are installed once at program seed time; the active rank is
// bound inside every graph body lambda so both the capture path and the
// non-graph fallback see the right device context. The two ranks' eager bodies
// are enqueued by two concurrent host threads (one per device), so the active
// rank is thread-local.
inline TpGroup* g_group        = nullptr;
inline thread_local int g_rank = 0;
// Per-rank, per-site fixed scratch (stable cudaMalloc'd addresses so
// graph-captured kernels keep valid pointers): delta[r][site] receives rank r's
// sharded projection output, sum[r][site] receives the reunited activation. The
// peer's delta pointer is a host-known constant, which the direct allreduce
// path reads through the P2P mapping.
inline void* g_delta[2][2]      = {{nullptr, nullptr}, {nullptr, nullptr}};
inline void* g_sum[2][2]        = {{nullptr, nullptr}, {nullptr, nullptr}};
inline void* g_peer_delta[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
inline std::int32_t g_site_rows       = 0; // hidden size (5120)
inline std::int32_t g_site_tokens_max = 0; // prefill chunk (>= verify width)
// Per-rank call counters: the staged-path parity slot must advance per rank
// (both ranks' k-th call at a site must pick the same slot index), and a single
// shared counter would interleave the two ranks' enqueues and desynchronize
// them.
inline std::size_t g_rank_site_calls[2][2] = {{0, 0}, {0, 0}};

inline bool active() noexcept { return g_group != nullptr; }
inline int rank() noexcept { return g_rank; }

// True on the rank that owns egress (sampling, lm_head, host read-back).
inline bool head_rank() noexcept { return !active() || g_rank == 0; }

inline void bind_rank(const DeviceContext& ctx) {
    if (g_group == nullptr) { return; }
    g_rank = (ctx.device == g_group->rank(0).device) ? 0 : 1;
}

inline Tensor site_delta(std::size_t site, std::int32_t tokens) {
    return Tensor(g_delta[g_rank][site], DType::BF16, {g_site_rows, tokens});
}

inline Tensor site_sum(std::size_t site, std::int32_t tokens) {
    return Tensor(g_sum[g_rank][site], DType::BF16, {g_site_rows, tokens});
}

// Reunites this rank's sharded projection (delta[g_rank][site]) with the peer's:
// sum[g_rank][site] = delta_rank0 + delta_rank1 (BF16). Both ranks must call
// the same site the same number of times, in lockstep order.
inline void allreduce_bf16(std::size_t site, std::int32_t tokens) {
    const std::int64_t count = static_cast<std::int64_t>(g_site_rows) * tokens;
    const std::size_t parity = g_rank_site_calls[g_rank][site] & 1;
    ++g_rank_site_calls[g_rank][site];
    g_group->allreduce_bf16_half(static_cast<std::size_t>(g_rank), g_delta[g_rank][site],
                                 g_peer_delta[g_rank][site], g_sum[g_rank][site], count, site,
                                 parity);
}

} // namespace ninfer::tpexec
