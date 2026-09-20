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
        *flag = generation;
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

TpGroup::TpGroup(std::vector<DeviceContext> contexts) : ranks_(std::move(contexts)) {
    if (ranks_.size() != kRankCount) {
        throw std::invalid_argument("TpGroup requires exactly two rank devices");
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
        ranks_[r].bind_to_current_thread();
        const cudaError_t counter_err =
            cudaMalloc(&counters_[r], kTpMaxSites * sizeof(unsigned long long));
        if (counter_err != cudaSuccess) {
            throw std::runtime_error(
                tp_cuda_error("TpGroup counter cudaMalloc failed", counter_err));
        }
        CUDA_CHECK(cudaMemsetAsync(counters_[r], 0, kTpMaxSites * sizeof(unsigned long long),
                                   ranks_[r].stream));
        CUDA_CHECK(cudaStreamSynchronize(ranks_[r].stream));
    }
}

TpGroup::~TpGroup() {
    for (std::size_t r = 0; r < kRankCount; ++r) {
        if (counters_[r] != nullptr) {
            ranks_[r].bind_to_current_thread_noexcept();
            static_cast<void>(cudaFree(counters_[r]));
            counters_[r] = nullptr;
        }
    }
    if (mailbox_ != nullptr) {
        static_cast<void>(cudaFreeHost(mailbox_));
        mailbox_ = nullptr;
    }
}

TpGroup::TpGroup(TpGroup&& other) noexcept : ranks_(std::move(other.ranks_)) {
    mailbox_ = std::exchange(other.mailbox_, nullptr);
    for (std::size_t r = 0; r < kRankCount; ++r) {
        counters_[r]       = other.counters_[r];
        other.counters_[r] = nullptr;
    }
}

TpGroup& TpGroup::operator=(TpGroup&& other) noexcept {
    if (this == &other) { return *this; }
    ranks_   = std::move(other.ranks_);
    mailbox_ = std::exchange(other.mailbox_, nullptr);
    for (std::size_t r = 0; r < kRankCount; ++r) {
        counters_[r]       = other.counters_[r];
        other.counters_[r] = nullptr;
    }
    return *this;
}

DeviceContext& TpGroup::rank(std::size_t index) {
    if (index >= ranks_.size()) { throw std::out_of_range("TpGroup rank index"); }
    return ranks_[index];
}

const DeviceContext& TpGroup::rank(std::size_t index) const {
    if (index >= ranks_.size()) { throw std::out_of_range("TpGroup rank index"); }
    return ranks_[index];
}

void TpGroup::enable_peer_access() {
    for (std::size_t r = 0; r < kRankCount; ++r) {
        DeviceContext& ctx    = ranks_[r];
        const int peer_device = ranks_[r ^ 1].device;
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
            DeviceContext& ctx = ranks_[r];
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
    // sum from HBM-local buffers behind the peer's generation.
    const std::size_t bytes = static_cast<std::size_t>(count) * 2;
    if (staging_[0] == nullptr || staging_bytes_ < bytes) {
        for (std::size_t r = 0; r < kRankCount; ++r) {
            ranks_[r].bind_to_current_thread();
            if (staging_[r] != nullptr) { CUDA_CHECK(cudaFree(staging_[r])); }
            CUDA_CHECK(cudaMalloc(&staging_[r], bytes));
        }
        staging_bytes_ = bytes;
    }
    for (std::size_t r = 0; r < kRankCount; ++r) {
        DeviceContext& ctx = ranks_[r];
        ctx.bind_to_current_thread();
        const int blocks = 8 * ctx.multiprocessor_count();
        tp_push_staging<<<blocks, kBlock, 0, ctx.stream>>>(
            static_cast<const uint4*>(inputs[r]),
            static_cast<uint4*>(staging_[r ^ 1]), count_vec8);
        CUDA_CHECK(cudaGetLastError());
    }
    for (std::size_t r = 0; r < kRankCount; ++r) {
        DeviceContext& ctx = ranks_[r];
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

} // namespace ninfer
