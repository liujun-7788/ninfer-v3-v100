// Standalone TpGroup smoke test: correctness of the two-rank BF16 allreduce (eager and
// CUDA-Graph replay), plus measured end-to-end latency at decode size and bandwidth at
// prefill size.
// Build: nvcc -O2 -std=c++20 -arch=sm_70 -Isrc tp_group_smoke.cu core/tp_group.cu core/device.cu
#include "core/device.h"
#include "core/tp_group.h"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using ninfer::DeviceContext;
using ninfer::TpGroup;

namespace {

struct PairBuffers {
    void* in[2]  = {nullptr, nullptr};
    void* out[2] = {nullptr, nullptr};
};

void upload(TpGroup& group, PairBuffers& bufs, std::int64_t count,
            const std::vector<std::vector<float>>& inputs) {
    for (int r = 0; r < 2; ++r) {
        group.rank(static_cast<std::size_t>(r)).bind_to_current_thread();
        std::vector<__nv_bfloat16> host(static_cast<std::size_t>(count));
        for (std::int64_t i = 0; i < count; ++i) {
            host[static_cast<std::size_t>(i)] =
                __float2bfloat16(inputs[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)]);
        }
        CUDA_CHECK(cudaMemcpy(bufs.in[r], host.data(), static_cast<std::size_t>(count) * 2,
                              cudaMemcpyHostToDevice));
    }
}

int verify(TpGroup& group, const PairBuffers& bufs, std::int64_t count,
           const std::vector<std::vector<float>>& inputs) {
    int failures = 0;
    for (int r = 0; r < 2; ++r) {
        std::vector<__nv_bfloat16> host(static_cast<std::size_t>(count));
        group.rank(static_cast<std::size_t>(r)).bind_to_current_thread();
        CUDA_CHECK(cudaMemcpy(host.data(), bufs.out[r], static_cast<std::size_t>(count) * 2,
                              cudaMemcpyDeviceToHost));
        for (std::int64_t i = 0; i < count; ++i) {
            const float got = __bfloat162float(host[static_cast<std::size_t>(i)]);
            const float want =
                inputs[0][static_cast<std::size_t>(i)] + inputs[1][static_cast<std::size_t>(i)];
            if (std::fabs(got - want) > 0.05f * std::fabs(want) + 0.05f) {
                if (failures < 5) {
                    std::printf("  MISMATCH rank=%d i=%lld got=%f want=%f\n", r,
                                static_cast<long long>(i), got, want);
                }
                ++failures;
            }
        }
    }
    return failures;
}

void sync_both(TpGroup& group) {
    for (int r = 0; r < 2; ++r) {
        group.rank(static_cast<std::size_t>(r)).bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(group.rank(static_cast<std::size_t>(r)).stream));
    }
}

float time_calls(TpGroup& group, const PairBuffers& bufs, std::int64_t count, std::size_t site,
                 int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) { group.allreduce_bf16(bufs.in, bufs.out, count, site); }
    sync_both(group);
    cudaEvent_t ev0;
    cudaEvent_t ev1;
    group.rank(0).bind_to_current_thread();
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0, group.rank(0).stream));
    for (int i = 0; i < iters; ++i) { group.allreduce_bf16(bufs.in, bufs.out, count, site); }
    CUDA_CHECK(cudaEventRecord(ev1, group.rank(0).stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    return ms / iters;
}

} // namespace

int main(int argc, char** argv) {
    const int dev0 = argc > 1 ? atoi(argv[1]) : 1;
    const int dev1 = argc > 2 ? atoi(argv[2]) : 2;

    std::vector<DeviceContext> contexts;
    contexts.emplace_back(dev0);
    contexts.emplace_back(dev1);
    TpGroup group(std::move(contexts));
    std::printf("TpGroup over (%d,%d)\n", dev0, dev1);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);

    const std::int64_t count = 5120 * 2048 + 8;
    std::vector<std::vector<float>> inputs(2);
    for (int r = 0; r < 2; ++r) {
        inputs[static_cast<std::size_t>(r)].resize(static_cast<std::size_t>(count));
        for (auto& v : inputs[static_cast<std::size_t>(r)]) { v = dist(rng); }
    }
    PairBuffers bufs;
    for (int r = 0; r < 2; ++r) {
        group.rank(static_cast<std::size_t>(r)).bind_to_current_thread();
        CUDA_CHECK(cudaMalloc(&bufs.in[r], static_cast<std::size_t>(count) * 2));
        CUDA_CHECK(cudaMalloc(&bufs.out[r], static_cast<std::size_t>(count) * 2));
    }

    // 1. Eager correctness across several sites.
    int failures = 0;
    for (std::size_t site : {std::size_t{0}, std::size_t{1}, std::size_t{2}}) {
        upload(group, bufs, count, inputs);
        group.allreduce_bf16(bufs.in, bufs.out, count, site);
        sync_both(group);
        failures += verify(group, bufs, count, inputs);
    }
    std::printf("eager correctness %s (failures=%d)\n", failures == 0 ? "PASS" : "FAIL", failures);

    // 2. CUDA Graph replay correctness: each rank captures its own two calls; the mailbox
    // handshake alone orders the ranks, so the two graphs share no host-side dependency.
    {
        upload(group, bufs, count, inputs);
        cudaGraph_t graph0 = nullptr;
        cudaGraph_t graph1 = nullptr;
        group.rank(0).bind_to_current_thread();
        CUDA_CHECK(cudaStreamBeginCapture(group.rank(0).stream, cudaStreamCaptureModeThreadLocal));
        group.allreduce_bf16(bufs.in, bufs.out, count, 0);
        group.allreduce_bf16(bufs.in, bufs.out, count, 1);
        CUDA_CHECK(cudaStreamEndCapture(group.rank(0).stream, &graph0));
        group.rank(1).bind_to_current_thread();
        CUDA_CHECK(cudaStreamBeginCapture(group.rank(1).stream, cudaStreamCaptureModeThreadLocal));
        group.allreduce_bf16(bufs.in, bufs.out, count, 0);
        group.allreduce_bf16(bufs.in, bufs.out, count, 1);
        CUDA_CHECK(cudaStreamEndCapture(group.rank(1).stream, &graph1));
        cudaGraphExec_t exec0 = nullptr;
        cudaGraphExec_t exec1 = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&exec0, graph0, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphInstantiate(&exec1, graph1, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            upload(group, bufs, count, inputs);
            group.rank(0).bind_to_current_thread();
            CUDA_CHECK(cudaGraphLaunch(exec0, group.rank(0).stream));
            group.rank(1).bind_to_current_thread();
            CUDA_CHECK(cudaGraphLaunch(exec1, group.rank(1).stream));
            sync_both(group);
            const int replay_failures = verify(group, bufs, count, inputs);
            failures += replay_failures;
            std::printf("graph replay %d: %s (failures=%d)\n", replay,
                        replay_failures == 0 ? "PASS" : "FAIL", replay_failures);
        }
        CUDA_CHECK(cudaGraphExecDestroy(exec0));
        CUDA_CHECK(cudaGraphExecDestroy(exec1));
        CUDA_CHECK(cudaGraphDestroy(graph0));
        CUDA_CHECK(cudaGraphDestroy(graph1));
    }

    // 3. Decode-size latency (1 x 5120 bf16), eager and under graph replay.
    {
        const std::int64_t small = 5120;
        const float per_call = time_calls(group, bufs, small, 0, 50, 500);
        std::printf("decode-size allreduce (5120 bf16) eager: %.1f us/call\n", per_call * 1e3f);

        cudaGraph_t g0 = nullptr;
        cudaGraph_t g1 = nullptr;
        group.rank(0).bind_to_current_thread();
        CUDA_CHECK(cudaStreamBeginCapture(group.rank(0).stream, cudaStreamCaptureModeThreadLocal));
        group.allreduce_bf16(bufs.in, bufs.out, small, 3);
        CUDA_CHECK(cudaStreamEndCapture(group.rank(0).stream, &g0));
        group.rank(1).bind_to_current_thread();
        CUDA_CHECK(cudaStreamBeginCapture(group.rank(1).stream, cudaStreamCaptureModeThreadLocal));
        group.allreduce_bf16(bufs.in, bufs.out, small, 3);
        CUDA_CHECK(cudaStreamEndCapture(group.rank(1).stream, &g1));
        cudaGraphExec_t e0 = nullptr;
        cudaGraphExec_t e1 = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&e0, g0, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphInstantiate(&e1, g1, nullptr, nullptr, 0));
        cudaEvent_t gev0 = nullptr;
        cudaEvent_t gev1 = nullptr;
        group.rank(0).bind_to_current_thread();
        CUDA_CHECK(cudaEventCreate(&gev0));
        CUDA_CHECK(cudaEventCreate(&gev1));
        CUDA_CHECK(cudaEventRecord(gev0, group.rank(0).stream));
        for (int i = 0; i < 100; ++i) {
            group.rank(0).bind_to_current_thread();
            CUDA_CHECK(cudaGraphLaunch(e0, group.rank(0).stream));
            group.rank(1).bind_to_current_thread();
            CUDA_CHECK(cudaGraphLaunch(e1, group.rank(1).stream));
        }
        CUDA_CHECK(cudaEventRecord(gev1, group.rank(0).stream));
        CUDA_CHECK(cudaEventSynchronize(gev1));
        float gms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&gms, gev0, gev1));
        std::printf("decode-size allreduce (5120 bf16) graph: %.1f us/call\n", gms / 100 * 1e3f);
        CUDA_CHECK(cudaEventDestroy(gev0));
        CUDA_CHECK(cudaEventDestroy(gev1));
        CUDA_CHECK(cudaGraphExecDestroy(e0));
        CUDA_CHECK(cudaGraphExecDestroy(e1));
        CUDA_CHECK(cudaGraphDestroy(g0));
        CUDA_CHECK(cudaGraphDestroy(g1));
    }

    // 4. Prefill-size (2048 x 5120 bf16).
    {
        const std::int64_t big = 2048 * 5120;
        const float per_call = time_calls(group, bufs, big, 1, 5, 50);
        std::printf("prefill-size allreduce (%lld bf16): %.3f ms/call (%.2f GB/s effective)\n",
                    static_cast<long long>(big), per_call,
                    static_cast<double>(big) * 2 * 2 / 1e9 / (per_call / 1e3));
    }

    for (int r = 0; r < 2; ++r) {
        group.rank(static_cast<std::size_t>(r)).bind_to_current_thread();
        CUDA_CHECK(cudaFree(bufs.in[r]));
        CUDA_CHECK(cudaFree(bufs.out[r]));
    }
    std::printf("%s\n", failures == 0 ? "SMOKE PASS" : "SMOKE FAIL");
    return failures == 0 ? 0 : 1;
}
