#pragma once

#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer {

inline constexpr std::size_t kPpMaxSites = 256;
inline constexpr std::size_t kPpMaxRanks = 8;

// Linear-pipeline link over peer-accessible GPUs. Ranks are ordered; data flows only
// between adjacent ranks (r pushes into r+1's staging). Every channel (one direction of
// one adjacent pair) has a monotone generation published in a host-pinned mailbox — PCIe
// coherence makes host memory the safe signaling medium, and one writer per channel word
// keeps values from regressing. Counters advance on device, keeping CUDA Graph replays in
// lockstep without host involvement.
//
// Channel roles are split: counter_tx counts a rank's outgoing publications, counter_rx
// counts its incoming waits. Both advance once per round per site, so each channel's
// generations stay in lockstep.
//
// Non-owning: the caller keeps every rank DeviceContext alive (the engine owns the
// primary; a bundle owning peer contexts may sit beside the link).
class PpLink {
public:
    explicit PpLink(std::vector<DeviceContext*> ranks);
    // Owning-peer convenience for the two-rank entry point (engine keeps rank0).
    PpLink(DeviceContext& rank0, int peer_device);
    ~PpLink();

    PpLink(const PpLink&)            = delete;
    PpLink& operator=(const PpLink&) = delete;
    PpLink(PpLink&& other) noexcept;
    PpLink& operator=(PpLink&& other) noexcept;

    [[nodiscard]] std::size_t rank_count() const noexcept { return ranks_.size(); }
    [[nodiscard]] DeviceContext& rank(std::size_t index);
    [[nodiscard]] const DeviceContext& rank(std::size_t index) const;

    // Enqueue on the producer's stream: copy its input into the next rank's staging and
    // publish the round on the (producer, site) channel.
    void push(std::size_t producer, std::size_t site, const void* source, void* consumer_staging,
              std::size_t bytes);

    // Enqueue on the consumer's stream: block until this round's push from the previous
    // rank is visible.
    void wait_input(std::size_t consumer, std::size_t site);

    // Enqueue on the rank's stream: publish a completion generation without a transfer
    // (reverse-direction acks for the decode turnstile).
    void signal_done(std::size_t rank, std::size_t site);

    // Enqueue on the rank's stream: block until the next rank's completion reaches this
    // round (decode turnstile; rank must not be the last).
    void wait_done(std::size_t rank, std::size_t site);

    // One-time setup for a wait site whose first round must pass without a preceding
    // publication (the decode turnstile head). Initializes the rank's receive counter so
    // its first expected generation is zero.
    void arm_zero_wait(std::size_t rank, std::size_t site);

private:
    struct Mailbox {
        unsigned long long flag[kPpMaxRanks][kPpMaxSites];
        unsigned long long counter_tx[kPpMaxRanks][kPpMaxSites];
        unsigned long long counter_rx[kPpMaxRanks][kPpMaxSites];
    };

    void enable_peer_access();
    void init_mailbox();

    std::vector<DeviceContext*> ranks_;
    Mailbox* mailbox_ = nullptr;
    std::unique_ptr<DeviceContext> peer_owner_;
};

} // namespace ninfer
