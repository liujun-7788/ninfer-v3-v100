#include "core/tp_group.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

std::string tp_cuda_error(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

constexpr std::int64_t kDirectPathMaxBytes = 512 * 1024;

// Signal kernel: one block per call publishes this round's generation to the host mailbox.
// The device counter keeps generations monotone and advancing under CUDA Graph replay; the
// system-scope fence orders the peer-visible input (produced by prior kernels on this
// stream) before the flag store. Exactly one writer per rank and site means the flag word
// never races.
__global__ void tp_signal(unsigned long long* counter, volatile unsigned long long* flag,
                          int site) {
    if (threadIdx.x == 0) {
        const unsigned long long generation = atomicAdd(counter + site, 1ULL) + 1ULL;
        __threadfence_system();
        flag[site] = generation;
    }
}

// Large path exchange: this rank pushes its full input into the peer's staging buffer with
// posted remote writes (no read round trips, and everything stays SM work so no
// copy-engine transitions are involved). The peer's staging is then HBM-local for the sum.
__global__ void tp_push_staging(const uint4* __restrict__ mine, uint4* __restrict__ peer_staging,
                                int count_vec8) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) *
                                static_cast<std::int64_t>(blockDim.x);
    const std::int64_t begin = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::int64_t i = begin; i < count_vec8; i += stride) { peer_staging[i] = mine[i]; }
}

__global__ void tp_allreduce_direct(const uint4* __restrict__ mine,
                                    const uint4* __restrict__ peer, uint4* __restrict__ out,
                                    const volatile unsigned long long* __restrict__ my_flag,
                                    const volatile unsigned long long* __restrict__ peer_flag,
                                    int site, int count_vec8) {
    if (threadIdx.x == 0) {
        const unsigned long long generation = my_flag[site];
        const auto* watch                   = peer_flag + site;
        while (*watch < generation) { __nanosleep(256); }
    }
    __syncthreads();
    __threadfence();

    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) *
                                static_cast<std::int64_t>(blockDim.x);
    const std::int64_t begin = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::int64_t i = begin; i < count_vec8; i += stride) {
        const uint4 a = mine[i];
        const uint4 b = peer[i];
        uint4 r;
        const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&a);
        const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&b);
        __nv_bfloat16* pr       = reinterpret_cast<__nv_bfloat16*>(&r);
        for (int k = 0; k < 8; ++k) {
            pr[k] = __float2bfloat16(__bfloat162float(pa[k]) + __bfloat162float(pb[k]));
        }
        out[i] = r;
    }
}

// Large-path sum: the peer half was pushed into local staging, so all traffic is HBM-local.
__global__ void tp_allreduce_staged(const uint4* __restrict__ mine,
                                    const uint4* __restrict__ staged, uint4* __restrict__ out,
                                    const volatile unsigned long long* __restrict__ my_flag,
                                    const volatile unsigned long long* __restrict__ peer_flag,
                                    int site, int count_vec8) {
    if (threadIdx.x == 0) {
        const unsigned long long generation = my_flag[site];
        const auto* watch                   = peer_flag + site;
        while (*watch < generation) { __nanosleep(256); }
    }
    __syncthreads();
    __threadfence();

    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) *
                                static_cast<std::int64_t>(blockDim.x);
    const std::int64_t begin = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::int64_t i = begin; i < count_vec8; i += stride) {
        const uint4 a = mine[i];
        const uint4 b = staged[i];
        uint4 r;
        const __nv_bfloat16* pa = reinterpret_cast<const __nv_bfloat16*>(&a);
        const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&b);
        __nv_bfloat16* pr       = reinterpret_cast<__nv_bfloat16*>(&r);
        for (int k = 0; k < 8; ++k) {
            pr[k] = __float2bfloat16(__bfloat162float(pa[k]) + __bfloat162float(pb[k]));
        }
        out[i] = r;
    }
}

} // namespace

TpGroup::TpGroup(DeviceContext& rank0, int peer_device, std::size_t staging_bytes) {
    ranks_[0]   = &rank0;
    peer_owner_ = std::make_unique<DeviceContext>(peer_device);
    ranks_[1]   = peer_owner_.get();
    if (ranks_[0]->device == ranks_[1]->device) {
        throw std::invalid_argument("TpGroup ranks must be distinct devices");
    }
    enable_peer_access();
    const cudaError_t err = cudaHostAlloc(&mailbox_, sizeof(Mailbox),
                                          cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        throw std::runtime_error(tp_cuda_error("TpGroup mailbox cudaHostAlloc failed", err));
    }
    std::fill(&mailbox_->flag[0][0],
              &mailbox_->flag[0][0] + sizeof(Mailbox) / sizeof(unsigned long long), 0ULL);
    for (std::size_t r = 0; r < kRankCount; ++r) {
        ranks_[r]->bind_to_current_thread();
        const cudaError_t counter_err =
            cudaMalloc(&counters_[r], kTpMaxSites * sizeof(unsigned long long));
        if (counter_err != cudaSuccess) {
            throw std::runtime_error(
                tp_cuda_error("TpGroup counter cudaMalloc failed", counter_err));
        }
        CUDA_CHECK(cudaMemsetAsync(counters_[r], 0, kTpMaxSites * sizeof(unsigned long long),
                                   ranks_[r]->stream));
        CUDA_CHECK(cudaStreamSynchronize(ranks_[r]->stream));
    }
    // Staged-path buffers must exist before any capture: cudaMalloc inside a
    // capturing stream is illegal, so the caller sizes them up front. Each
    // rank holds kTpStagedSites * kStagingParities slots of staging_bytes.
    staging_bytes_ = staging_bytes;
    if (staging_bytes_ != 0) {
        for (std::size_t r = 0; r < kRankCount; ++r) {
            ranks_[r]->bind_to_current_thread();
            const cudaError_t staging_err = cudaMalloc(
                &staging_[r], kTpStagedSites * kStagingParities * staging_bytes_);
            if (staging_err != cudaSuccess) {
                throw std::runtime_error(
                    tp_cuda_error("TpGroup staging cudaMalloc failed", staging_err));
            }
        }
    }
}

void TpGroup::allocate_staging(std::size_t staging_bytes) {
    if (staging_bytes_ != 0) {
        if (staging_bytes_ < staging_bytes) {
            throw std::invalid_argument(
                "TpGroup staging already allocated with a smaller size");
        }
        return;
    }
    if (staging_bytes == 0) {
        throw std::invalid_argument("TpGroup staging bytes must be positive");
    }
    staging_bytes_ = staging_bytes;
    for (std::size_t r = 0; r < kRankCount; ++r) {
        ranks_[r]->bind_to_current_thread();
        const cudaError_t staging_err =
            cudaMalloc(&staging_[r], kTpStagedSites * kStagingParities * staging_bytes_);
        if (staging_err != cudaSuccess) {
            throw std::runtime_error(
                tp_cuda_error("TpGroup staging cudaMalloc failed", staging_err));
        }
    }
}

TpGroup::~TpGroup() {
    for (std::size_t r = 0; r < kRankCount; ++r) {
        if (ranks_[r] != nullptr) { ranks_[r]->bind_to_current_thread_noexcept(); }
        if (staging_[r] != nullptr) {
            static_cast<void>(cudaFree(staging_[r]));
            staging_[r] = nullptr;
        }
        if (counters_[r] != nullptr && ranks_[r] != nullptr) {
            static_cast<void>(cudaFree(counters_[r]));
            counters_[r] = nullptr;
        }
    }
    if (mailbox_ != nullptr) {
        static_cast<void>(cudaFreeHost(mailbox_));
        mailbox_ = nullptr;
    }
}

TpGroup::TpGroup(TpGroup&& other) noexcept
    : peer_owner_(std::move(other.peer_owner_)) {
    ranks_[0] = std::exchange(other.ranks_[0], nullptr);
    ranks_[1] = std::exchange(other.ranks_[1], nullptr);
    mailbox_  = std::exchange(other.mailbox_, nullptr);
    for (std::size_t r = 0; r < kRankCount; ++r) {
        counters_[r]       = other.counters_[r];
        other.counters_[r] = nullptr;
        staging_[r]        = std::exchange(other.staging_[r], nullptr);
    }
    staging_bytes_ = std::exchange(other.staging_bytes_, 0);
}

TpGroup& TpGroup::operator=(TpGroup&& other) noexcept {
    if (this == &other) { return *this; }
    ranks_[0]   = std::exchange(other.ranks_[0], nullptr);
    ranks_[1]   = std::exchange(other.ranks_[1], nullptr);
    peer_owner_ = std::move(other.peer_owner_);
    mailbox_    = std::exchange(other.mailbox_, nullptr);
    for (std::size_t r = 0; r < kRankCount; ++r) {
        counters_[r]       = other.counters_[r];
        other.counters_[r] = nullptr;
        staging_[r]        = std::exchange(other.staging_[r], nullptr);
    }
    staging_bytes_ = std::exchange(other.staging_bytes_, 0);
    return *this;
}

DeviceContext& TpGroup::rank(std::size_t index) {
    if (index >= kRankCount || ranks_[index] == nullptr) {
        throw std::out_of_range("TpGroup rank index");
    }
    return *ranks_[index];
}

const DeviceContext& TpGroup::rank(std::size_t index) const {
    if (index >= kRankCount || ranks_[index] == nullptr) {
        throw std::out_of_range("TpGroup rank index");
    }
    return *ranks_[index];
}

void TpGroup::enable_peer_access() {
    for (std::size_t r = 0; r < kRankCount; ++r) {
        DeviceContext& ctx    = *ranks_[r];
        const int peer_device = ranks_[r ^ 1]->device;
        int accessible        = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, ctx.device, peer_device));
        if (accessible == 0) {
            throw std::runtime_error("TpGroup ranks have no P2P peer access path");
        }
        ctx.bind_to_current_thread();
        const cudaError_t err = cudaDeviceEnablePeerAccess(peer_device, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
            static_cast<void>(cudaGetLastError());
            continue;
        }
        if (err != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            throw std::runtime_error(tp_cuda_error("cudaDeviceEnablePeerAccess failed", err));
        }
    }
}

void TpGroup::allreduce_bf16(const void* const inputs[kRankCount],
                             void* const outputs[kRankCount], std::int64_t count,
                             std::size_t site) {
    if (count <= 0) { return; }
    if (site >= kTpMaxSites) { throw std::out_of_range("TpGroup site index"); }
    if ((count & 7) != 0) {
        throw std::invalid_argument("TpGroup allreduce count must be a multiple of 8");
    }
    for (std::size_t r = 0; r < kRankCount; ++r) {
        if (inputs[r] == nullptr || outputs[r] == nullptr) {
            throw std::invalid_argument("TpGroup allreduce buffers must be device memory");
        }
        if (inputs[r] == outputs[r]) {
            throw std::invalid_argument("TpGroup allreduce output must not alias its input");
        }
    }

    const int count_vec8 = static_cast<int>(count / 8);
    constexpr int kBlock = 256;
    if (count * 2 <= kDirectPathMaxBytes) {
    for (std::size_t r = 0; r < kRankCount; ++r) {
        DeviceContext& ctx = *ranks_[r];
        ctx.bind_to_current_thread();
        // Same-stream ordering is load-bearing: the signal publishes data that prior
        // kernels on this stream produced, so it must be enqueued behind them.
            tp_signal<<<1, 32, 0, ctx.stream>>>(counters_[r], mailbox_->flag[r],
                                                static_cast<int>(site));
            CUDA_CHECK(cudaGetLastError());
            const std::int64_t wanted = (count_vec8 + kBlock - 1) / kBlock;
            const int blocks = static_cast<int>(std::min<std::int64_t>(
                std::max<std::int64_t>(wanted, 1), 8LL * ctx.multiprocessor_count()));
            tp_allreduce_direct<<<blocks, kBlock, 0, ctx.stream>>>(
                static_cast<const uint4*>(inputs[r]), static_cast<const uint4*>(inputs[r ^ 1]),
                static_cast<uint4*>(outputs[r]), mailbox_->flag[r], mailbox_->flag[r ^ 1],
                static_cast<int>(site), count_vec8);
            CUDA_CHECK(cudaGetLastError());
        }
        return;
    }
    // Large path: push the full input into the peer's staging with posted remote writes
    // (SM work at line rate, no copy-engine transitions, graph capturable), publish, then
    // sum from HBM-local buffers behind the peer's generation. Staging is preallocated at
    // construction (cudaMalloc is illegal inside a capturing stream); single-slot use is
    // safe here because this entry point pairs rank-by-rank under one caller thread that
    // keeps both streams in lockstep across the whole collective.
    const std::size_t bytes = static_cast<std::size_t>(count) * 2;
    if (staging_[0] == nullptr || staging_bytes_ < bytes) {
        throw std::invalid_argument(
            "TpGroup staged allreduce exceeds the preallocated staging size");
    }
    for (std::size_t r = 0; r < kRankCount; ++r) {
        DeviceContext& ctx = *ranks_[r];
        ctx.bind_to_current_thread();
        const int blocks = 8 * ctx.multiprocessor_count();
        tp_push_staging<<<blocks, kBlock, 0, ctx.stream>>>(
            static_cast<const uint4*>(inputs[r]),
            static_cast<uint4*>(staging_[r ^ 1]), count_vec8);
        CUDA_CHECK(cudaGetLastError());
    }
    for (std::size_t r = 0; r < kRankCount; ++r) {
        DeviceContext& ctx = *ranks_[r];
        ctx.bind_to_current_thread();
        tp_signal<<<1, 32, 0, ctx.stream>>>(counters_[r], mailbox_->flag[r],
                                            static_cast<int>(site));
        CUDA_CHECK(cudaGetLastError());
        const std::int64_t wanted = (count_vec8 + kBlock - 1) / kBlock;
        const int blocks = static_cast<int>(std::min<std::int64_t>(
            std::max<std::int64_t>(wanted, 1), 8LL * ctx.multiprocessor_count()));
        tp_allreduce_staged<<<blocks, kBlock, 0, ctx.stream>>>(
            static_cast<const uint4*>(inputs[r]), static_cast<const uint4*>(staging_[r]),
            static_cast<uint4*>(outputs[r]), mailbox_->flag[r], mailbox_->flag[r ^ 1],
            static_cast<int>(site), count_vec8);
        CUDA_CHECK(cudaGetLastError());
    }
}

void TpGroup::allreduce_bf16_half(std::size_t r, const void* input, const void* peer_input,
                                  void* output, std::int64_t count, std::size_t site,
                                  std::size_t parity) {
    if (r >= kRankCount) { throw std::out_of_range("TpGroup rank index"); }
    if (count <= 0) { return; }
    if (site >= kTpMaxSites) { throw std::out_of_range("TpGroup site index"); }
    if ((count & 7) != 0) {
        throw std::invalid_argument("TpGroup allreduce count must be a multiple of 8");
    }
    if (input == nullptr || peer_input == nullptr || output == nullptr) {
        throw std::invalid_argument("TpGroup allreduce buffers must be device memory");
    }
    if (input == output) {
        throw std::invalid_argument("TpGroup allreduce output must not alias its input");
    }

    DeviceContext& ctx = *ranks_[r];
    const int count_vec8 = static_cast<int>(count / 8);
    constexpr int kBlock = 256;
    if (count * 2 <= kDirectPathMaxBytes) {
        ctx.bind_to_current_thread();
        // Same-stream ordering is load-bearing: the signal publishes data that prior
        // kernels on this stream produced, so it must be enqueued behind them.
        tp_signal<<<1, 32, 0, ctx.stream>>>(counters_[r], mailbox_->flag[r],
                                            static_cast<int>(site));
        CUDA_CHECK(cudaGetLastError());
        const std::int64_t wanted = (count_vec8 + kBlock - 1) / kBlock;
        const int blocks = static_cast<int>(std::min<std::int64_t>(
            std::max<std::int64_t>(wanted, 1), 8LL * ctx.multiprocessor_count()));
        tp_allreduce_direct<<<blocks, kBlock, 0, ctx.stream>>>(
            static_cast<const uint4*>(input), static_cast<const uint4*>(peer_input),
            static_cast<uint4*>(output), mailbox_->flag[r], mailbox_->flag[r ^ 1],
            static_cast<int>(site), count_vec8);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // Staged path: push this rank's input into the peer's staging slot, publish, then sum
    // from the peer's pushed half sitting in this rank's own HBM. The parity slot alternates
    // per call so call i+1's push cannot overwrite the staging buffer a still-running sum(i)
    // reads: the push of call i+2 is stream-ordered behind this rank's sum(i+1), which
    // waits for the peer's signal(i+1), which is stream-ordered behind the peer's sum(i)
    // completion.
    const std::size_t bytes = static_cast<std::size_t>(count) * 2;
    if (staging_[r] == nullptr || staging_bytes_ < bytes) {
        throw std::invalid_argument(
            "TpGroup staged allreduce exceeds the preallocated staging size");
    }
    if (site >= kTpStagedSites || parity >= kStagingParities) {
        throw std::out_of_range("TpGroup staged allreduce site/parity");
    }
    const std::size_t slot_bytes =
        site * kStagingParities * staging_bytes_ + parity * staging_bytes_;
    auto* peer_slot  = static_cast<unsigned char*>(staging_[r ^ 1]) + slot_bytes;
    auto* my_slot    = static_cast<unsigned char*>(staging_[r]) + slot_bytes;
    ctx.bind_to_current_thread();
    const int blocks = 8 * ctx.multiprocessor_count();
    tp_push_staging<<<blocks, kBlock, 0, ctx.stream>>>(static_cast<const uint4*>(input),
                                                       reinterpret_cast<uint4*>(peer_slot),
                                                       count_vec8);
    CUDA_CHECK(cudaGetLastError());
    tp_signal<<<1, 32, 0, ctx.stream>>>(counters_[r], mailbox_->flag[r],
                                        static_cast<int>(site));
    CUDA_CHECK(cudaGetLastError());
    const std::int64_t wanted = (count_vec8 + kBlock - 1) / kBlock;
    const int sum_blocks = static_cast<int>(std::min<std::int64_t>(
        std::max<std::int64_t>(wanted, 1), 8LL * ctx.multiprocessor_count()));
    tp_allreduce_staged<<<sum_blocks, kBlock, 0, ctx.stream>>>(
        static_cast<const uint4*>(input), reinterpret_cast<const uint4*>(my_slot),
        static_cast<uint4*>(output), mailbox_->flag[r], mailbox_->flag[r ^ 1],
        static_cast<int>(site), count_vec8);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer