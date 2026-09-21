// PpLink smoke: push/wait correctness (incl. zero-armed turnstile), 21MB push latency,
// and graph-replay correctness.
#include "core/device.h"
#include "core/pp_link.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using ninfer::DeviceContext;
using ninfer::PpLink;

int main(int argc, char** argv) {
    const int dev0 = argc > 1 ? atoi(argv[1]) : 0;
    const int dev1 = argc > 2 ? atoi(argv[2]) : 1;

    DeviceContext ctx0(dev0);
    DeviceContext ctx1(dev1);
    PpLink link(ctx0, ctx1);
    std::printf("PpLink over (%d,%d)\n", dev0, dev1);

    const std::size_t bytes = 21u << 20;
    void* src = nullptr;
    void* dst = nullptr;
    link.rank(0).bind_to_current_thread();
    CUDA_CHECK(cudaMalloc(&src, bytes));
    link.rank(1).bind_to_current_thread();
    CUDA_CHECK(cudaMalloc(&dst, bytes));
    CUDA_CHECK(cudaMemset(dst, 0, bytes));

    std::mt19937 rng(99);
    std::vector<unsigned int> pattern(bytes / 4);
    for (auto& v : pattern) { v = rng(); }
    link.rank(0).bind_to_current_thread();
    CUDA_CHECK(cudaMemcpy(src, pattern.data(), bytes, cudaMemcpyHostToDevice));

    // Round 1: push then wait then verify.
    link.push(0, src, dst, bytes, 0);
    link.wait(1, 0);
    link.rank(1).bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(link.rank(1).stream));
    std::vector<unsigned int> got(bytes / 4);
    CUDA_CHECK(cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost));
    const int mismatches = std::memcmp(got.data(), pattern.data(), bytes) != 0 ? 1 : 0;
    std::printf("round1 correctness %s\n", mismatches == 0 ? "PASS" : "FAIL");

    // Round 2 without a fresh push: wait must block until round 2's push (tests the
    // generation gating).
    link.push(0, src, dst, bytes, 0);
    link.wait(1, 0);
    link.rank(1).bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(link.rank(1).stream));
    std::printf("round2 gating PASS\n");

    // Zero-armed turnstile: a fresh site's wait passes before any push.
    link.arm_zero_wait(1, 1);
    link.wait(1, 1);
    link.rank(1).bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(link.rank(1).stream));
    std::printf("arm_zero_wait PASS\n");

    // 21MB push latency (steady state).
    for (int i = 0; i < 5; ++i) {
        link.push(0, src, dst, bytes, 2);
        link.wait(1, 2);
    }
    link.rank(0).bind_to_current_thread();
    link.rank(1).bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(link.rank(0).stream));
    CUDA_CHECK(cudaStreamSynchronize(link.rank(1).stream));
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    link.rank(0).bind_to_current_thread();
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    const int iters = 30;
    CUDA_CHECK(cudaEventRecord(ev0, link.rank(0).stream));
    for (int i = 0; i < iters; ++i) {
        link.push(0, src, dst, bytes, 2);
        link.wait(1, 2);
    }
    CUDA_CHECK(cudaEventRecord(ev1, link.rank(0).stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    std::printf("21MB push+wait: %.3f ms/round (%.2f GB/s)\n", ms / iters,
                static_cast<double>(bytes) / 1e9 / (ms / iters / 1e3));

    // Graph replay: capture push+wait on both sides, replay twice, verify.
    {
        cudaGraph_t g0 = nullptr;
        cudaGraph_t g1 = nullptr;
        link.rank(0).bind_to_current_thread();
        CUDA_CHECK(cudaStreamBeginCapture(link.rank(0).stream, cudaStreamCaptureModeThreadLocal));
        link.push(0, src, dst, bytes, 3);
        CUDA_CHECK(cudaStreamEndCapture(link.rank(0).stream, &g0));
        link.rank(1).bind_to_current_thread();
        CUDA_CHECK(cudaStreamBeginCapture(link.rank(1).stream, cudaStreamCaptureModeThreadLocal));
        link.wait(1, 3);
        CUDA_CHECK(cudaStreamEndCapture(link.rank(1).stream, &g1));
        cudaGraphExec_t e0 = nullptr;
        cudaGraphExec_t e1 = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&e0, g0, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphInstantiate(&e1, g1, nullptr, nullptr, 0));
        int graph_failures = 0;
        for (int replay = 0; replay < 2; ++replay) {
            link.rank(0).bind_to_current_thread();
            CUDA_CHECK(cudaGraphLaunch(e0, link.rank(0).stream));
            link.rank(1).bind_to_current_thread();
            CUDA_CHECK(cudaGraphLaunch(e1, link.rank(1).stream));
            link.rank(1).bind_to_current_thread();
            CUDA_CHECK(cudaStreamSynchronize(link.rank(1).stream));
            CUDA_CHECK(cudaMemcpy(got.data(), dst, bytes, cudaMemcpyDeviceToHost));
            if (std::memcmp(got.data(), pattern.data(), bytes) != 0) { ++graph_failures; }
        }
        std::printf("graph replay %s\n", graph_failures == 0 ? "PASS" : "FAIL");
        CUDA_CHECK(cudaGraphExecDestroy(e0));
        CUDA_CHECK(cudaGraphExecDestroy(e1));
        CUDA_CHECK(cudaGraphDestroy(g0));
        CUDA_CHECK(cudaGraphDestroy(g1));
    }

    CUDA_CHECK(cudaFree(src));
    link.rank(1).bind_to_current_thread();
    CUDA_CHECK(cudaFree(dst));
    std::printf("PP SMOKE %s\n", mismatches == 0 ? "PASS" : "FAIL");
    return mismatches == 0 ? 0 : 1;
}
