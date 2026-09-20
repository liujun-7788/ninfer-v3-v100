#pragma once

#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer {

inline constexpr std::size_t kPpMaxSites = 64;

// One-directional hidden-state link between two pipeline ranks over peer access. The
// producer pushes a buffer into the consumer's staging area (posted remote writes) and
// publishes a generation to a host-pinned mailbox; the consumer's stream gates behind that
// generation with a wait kernel. Both counters advance once per round on their own device,
// so the same launch sequence stays correct under CUDA Graph replay.
//
// The link owns no model state. All buffers are caller-provided; the consumer staging
// buffer lives on the consumer's device and must stay stable across graph captures.
class PpLink {
public:
    static constexpr std::size_t kRankCount = 2;

    explicit PpLink(std::vector<DeviceContext> contexts);
    ~PpLink();

    PpLink(const PpLink&)            = delete;
    PpLink& operator=(const PpLink&) = delete;
    PpLink(PpLink&& other) noexcept;
    PpLink& operator=(PpLink&& other) noexcept;

    [[nodiscard]] DeviceContext& rank(std::size_t index);
    [[nodiscard]] const DeviceContext& rank(std::size_t index) const;

    // Enqueue on the producer's stream: copy source (producer device) into destination
    // (consumer device) and publish the round. Consumer staging may be reallocated between
    // rounds but must be stable within one capture.
    void push(std::size_t producer, const void* source, void* consumer_staging,
              std::size_t bytes, std::size_t site);

    // Enqueue on the consumer's stream: block until this round's push is visible.
    void wait(std::size_t consumer, std::size_t site);

    // One-time setup for a wait site whose first round must pass without a preceding push
    // (the decode turnstile: rank0 round N+1 gates on rank1's round-N completion signal,
    // so the very first round must fall through). Initializes the consumer's counter so its
    // first expected generation is zero.
    void arm_zero_wait(std::size_t consumer, std::size_t site);

private:
    struct Mailbox {
        unsigned long long flag[kRankCount][kPpMaxSites];
    };

    void enable_peer_access();

    std::vector<DeviceContext> ranks_;
    Mailbox* mailbox_                         = nullptr;
    unsigned long long* counters_[kRankCount] = {nullptr, nullptr};
};

} // namespace ninfer
