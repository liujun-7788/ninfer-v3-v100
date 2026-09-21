#include "core/pp_link.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

std::string pp_cuda_error(const char* prefix, cudaError_t err) {
    return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

__global__ void pp_push_kernel(const uint4* __restrict__ source, uint4* __restrict__ staging,
                               long long count_vec8) {
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) *
                                static_cast<std::int64_t>(blockDim.x);
    const std::int64_t begin = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::int64_t i = begin; i < count_vec8; i += stride) { staging[i] = source[i]; }
}

// Producer side: publish this round's generation after the push drained past the point of
// peer coherence. One writer per rank and site keeps the flag monotone.
__global__ void pp_signal_kernel(unsigned long long* counter,
                                 volatile unsigned long long* flag) {
    if (threadIdx.x == 0) {
        const unsigned long long generation = atomicAdd(counter, 1ULL) + 1ULL;
        __threadfence_system();
        *flag = generation;
    }
}

// Consumer side: this round's generation comes from the consumer's own counter, which
// advances in lockstep with the producer's (one push and one wait per round per site).
__global__ void pp_wait_kernel(unsigned long long* counter,
                               const volatile unsigned long long* producer_flag) {
    if (threadIdx.x == 0) {
        const unsigned long long generation = atomicAdd(counter, 1ULL) + 1ULL;
        const auto* watch                   = producer_flag;
        while (*watch < generation) { __nanosleep(256); }
    }
}

} // namespace

void PpLink::init_mailbox() {
    const cudaError_t err = cudaHostAlloc(&mailbox_, sizeof(Mailbox),
                                          cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        throw std::runtime_error(pp_cuda_error("PpLink mailbox cudaHostAlloc failed", err));
    }
    std::fill(&mailbox_->flag[0][0],
              &mailbox_->flag[0][0] + sizeof(Mailbox) / sizeof(unsigned long long), 0ULL);
    for (std::size_t r = 0; r < kRankCount; ++r) {
        ranks_[r]->bind_to_current_thread();
        const cudaError_t counter_err =
            cudaMalloc(&counters_[r], kPpMaxSites * sizeof(unsigned long long));
        if (counter_err != cudaSuccess) {
            throw std::runtime_error(pp_cuda_error("PpLink counter cudaMalloc failed", counter_err));
        }
        CUDA_CHECK(cudaMemsetAsync(counters_[r], 0, kPpMaxSites * sizeof(unsigned long long),
                                   ranks_[r]->stream));
        CUDA_CHECK(cudaStreamSynchronize(ranks_[r]->stream));
    }
}

PpLink::PpLink(DeviceContext& rank0, DeviceContext& rank1) {
    ranks_[0] = &rank0;
    ranks_[1] = &rank1;
    if (ranks_[0]->device == ranks_[1]->device) {
        throw std::invalid_argument("PpLink requires two distinct devices");
    }
    enable_peer_access();
    init_mailbox();
}

PpLink::PpLink(DeviceContext& rank0, int peer_device)
    : peer_owner_(std::make_unique<DeviceContext>(peer_device)) {
    ranks_[0] = &rank0;
    ranks_[1] = peer_owner_.get();
    enable_peer_access();
    init_mailbox();
}

PpLink::~PpLink() {
    for (std::size_t r = 0; r < kRankCount; ++r) {
        if (counters_[r] != nullptr) {
            ranks_[r]->bind_to_current_thread_noexcept();
            static_cast<void>(cudaFree(counters_[r]));
            counters_[r] = nullptr;
        }
    }
    if (mailbox_ != nullptr) {
        static_cast<void>(cudaFreeHost(mailbox_));
        mailbox_ = nullptr;
    }
}

PpLink::PpLink(PpLink&& other) noexcept
    : ranks_(other.ranks_), mailbox_(std::exchange(other.mailbox_, nullptr)) {
    for (std::size_t r = 0; r < kRankCount; ++r) {
        counters_[r] = std::exchange(other.counters_[r], nullptr);
    }
}

PpLink& PpLink::operator=(PpLink&& other) noexcept {
    if (this == &other) { return *this; }
    ranks_   = other.ranks_;
    mailbox_ = std::exchange(other.mailbox_, nullptr);
    for (std::size_t r = 0; r < kRankCount; ++r) {
        counters_[r] = std::exchange(other.counters_[r], nullptr);
    }
    return *this;
}

DeviceContext& PpLink::rank(std::size_t index) {
    if (index >= kRankCount || ranks_[index] == nullptr) {
        throw std::out_of_range("PpLink rank index");
    }
    return *ranks_[index];
}

const DeviceContext& PpLink::rank(std::size_t index) const {
    if (index >= kRankCount || ranks_[index] == nullptr) {
        throw std::out_of_range("PpLink rank index");
    }
    return *ranks_[index];
}

void PpLink::enable_peer_access() {
    for (std::size_t r = 0; r < kRankCount; ++r) {
        DeviceContext& ctx    = *ranks_[r];
        const int peer_device = ranks_[r ^ 1]->device;
        int accessible        = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&accessible, ctx.device, peer_device));
        if (accessible == 0) {
            throw std::runtime_error("PpLink ranks have no P2P peer access path");
        }
        ctx.bind_to_current_thread();
        const cudaError_t err = cudaDeviceEnablePeerAccess(peer_device, 0);
        if (err == cudaErrorPeerAccessAlreadyEnabled) {
            static_cast<void>(cudaGetLastError());
            continue;
        }
        if (err != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            throw std::runtime_error(pp_cuda_error("cudaDeviceEnablePeerAccess failed", err));
        }
    }
}

void PpLink::push(std::size_t producer, const void* source, void* consumer_staging,
                  std::size_t bytes, std::size_t site) {
    if (producer >= kRankCount || site >= kPpMaxSites) {
        throw std::out_of_range("PpLink push index");
    }
    if (bytes == 0) { return; }
    if ((bytes & 15) != 0) {
        throw std::invalid_argument("PpLink push bytes must be a multiple of 16");
    }
    DeviceContext& ctx = *ranks_[producer];
    ctx.bind_to_current_thread();
    const long long count_vec8 = static_cast<long long>(bytes / 16);
    constexpr int kBlock       = 256;
    const long long wanted     = (count_vec8 + kBlock - 1) / kBlock;
    const int blocks = static_cast<int>(std::min<std::int64_t>(std::max<long long>(wanted, 1),
                                                               8LL * ctx.multiprocessor_count()));
    pp_push_kernel<<<blocks, kBlock, 0, ctx.stream>>>(
        static_cast<const uint4*>(source), static_cast<uint4*>(consumer_staging), count_vec8);
    CUDA_CHECK(cudaGetLastError());
    pp_signal_kernel<<<1, 32, 0, ctx.stream>>>(counters_[producer] + site,
                                               mailbox_->flag[producer] + site);
    CUDA_CHECK(cudaGetLastError());
}

void PpLink::wait(std::size_t consumer, std::size_t site) {
    if (consumer >= kRankCount || site >= kPpMaxSites) {
        throw std::out_of_range("PpLink wait index");
    }
    DeviceContext& ctx = *ranks_[consumer];
    ctx.bind_to_current_thread();
    pp_wait_kernel<<<1, 32, 0, ctx.stream>>>(counters_[consumer] + site,
                                             mailbox_->flag[consumer ^ 1] + site);
    CUDA_CHECK(cudaGetLastError());
}

void PpLink::arm_zero_wait(std::size_t consumer, std::size_t site) {
    if (consumer >= kRankCount || site >= kPpMaxSites) {
        throw std::out_of_range("PpLink arm index");
    }
    ranks_[consumer]->bind_to_current_thread();
    const unsigned long long minus_one = ~0ULL;
    CUDA_CHECK(cudaMemcpy(counters_[consumer] + site, &minus_one, sizeof(minus_one),
                          cudaMemcpyHostToDevice));
}

void PpLink::signal(std::size_t rank, std::size_t site) {
    if (rank >= kRankCount || site >= kPpMaxSites) {
        throw std::out_of_range("PpLink signal index");
    }
    DeviceContext& ctx = *ranks_[rank];
    ctx.bind_to_current_thread();
    pp_signal_kernel<<<1, 32, 0, ctx.stream>>>(counters_[rank] + site,
                                               mailbox_->flag[rank] + site);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer
