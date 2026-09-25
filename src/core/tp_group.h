#pragma once

#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace ninfer {

inline constexpr std::size_t kTpMaxSites = 256;

// Fixed two-rank peer group for tensor-parallel execution over P2P-accessible GPUs.
// The engine-level entry point takes the primary DeviceContext by reference (rank 0 stays
// the Engine's primary); the peer DeviceContext is created and owned by the group.
// Construction enables peer access in both directions; afterwards kernels on one rank may
// dereference device pointers owned by the other rank directly.
//
// The group owns no program state. It provides only the symmetric two-rank BF16 sum.
// Synchronization is a device-side generation mailbox incremented inside the collective
// kernel, so the same launch sequence stays correct under CUDA Graph replay.
//
// Each call launches exactly one kernel per rank: every rank reads both inputs (the peer's
// input through the peer mapping) and writes the full sum to its own output. Inputs and
// outputs must not alias. Site indexes select independent mailboxes so concurrent capture
// or replay of different call sites cannot cross-signal; a site must be used alternately
// by both ranks symmetrically, which every caller satisfies by construction.
class TpGroup {
public:
    static constexpr std::size_t kRankCount = 2;

    explicit TpGroup(DeviceContext& rank0, int peer_device, std::size_t staging_bytes = 0);
    ~TpGroup();

    TpGroup(const TpGroup&)            = delete;
    TpGroup& operator=(const TpGroup&) = delete;
    TpGroup(TpGroup&& other) noexcept;
    TpGroup& operator=(TpGroup&& other) noexcept;

    [[nodiscard]] std::size_t rank_count() const noexcept { return kRankCount; }
    [[nodiscard]] DeviceContext& rank(std::size_t index);
    [[nodiscard]] const DeviceContext& rank(std::size_t index) const;

    // Symmetric two-rank BF16 elementwise sum: outputs[r] = inputs[0] + inputs[1] on rank r.
    // inputs[r]/outputs[r] must be device memory on rank r; inputs and outputs must not alias.
    void allreduce_bf16(const void* const inputs[kRankCount], void* const outputs[kRankCount],
                        std::int64_t count, std::size_t site);

    // Single-rank entry point: enqueues one rank's half of the sum on that rank's own stream.
    // Each rank calls this independently (lockstep order, same site); the call enqueues the
    // signal and sum kernels on rank r's stream and returns immediately, so callers drive
    // both ranks by sequential enqueue without thread rebinding. input/output must be device
    // memory on rank r; peer_input must be device memory on rank r^1. parity alternates the
    // staged-path slot so a subsequent call cannot overwrite staging a still-running sum
    // reads (parity must match between ranks for the same call index).
    void allreduce_bf16_half(std::size_t r, const void* input, const void* peer_input,
                             void* output, std::int64_t count, std::size_t site,
                             std::size_t parity);

    // Preallocates the staged-path buffers. Must be called before any CUDA graph
    // capture (cudaMalloc is illegal inside a capturing stream) and at most once;
    // pass the largest single allreduce in bytes.
    void allocate_staging(std::size_t staging_bytes);

    [[nodiscard]] std::size_t staging_bytes() const noexcept { return staging_bytes_; }

private:
    // Host-pinned mailbox shared by both ranks: PCIe coherence makes it a safe signaling
    // medium where remote-device-memory spin is not (no NVLink on this topology). Each rank
    // has its own counter (device memory, incremented by its signal kernel) and its own
    // flag word (host mailbox, written only by that rank's signal kernel, so the value is
    // monotone within a site and racing stores cannot regress it).
    struct Mailbox {
        unsigned long long flag[kRankCount][kTpMaxSites];
    };

    void enable_peer_access();

    DeviceContext* ranks_[kRankCount]         = {nullptr, nullptr};
    std::unique_ptr<DeviceContext> peer_owner_;
    Mailbox* mailbox_                         = nullptr;
    unsigned long long* counters_[kRankCount] = {nullptr, nullptr};
    void* staging_[kRankCount]                = {nullptr, nullptr};
    std::size_t staging_bytes_                = 0;
    // Two parity slots per site in the staged path (slot = site*2 + parity),
    // allocated as one buffer per rank of 2 * staging_bytes.
    static constexpr std::size_t kStagingParities = 2;
    static constexpr std::size_t kTpStagedSites   = 2;
};

} // namespace ninfer
