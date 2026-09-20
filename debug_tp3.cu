// Compare memcpy engines: NULL stream vs non-blocking stream, peer copy 21MB.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x)                                                                \
    do {                                                                        \
        cudaError_t e = (x);                                                    \
        if (e != cudaSuccess) {                                                 \
            printf("ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main(int argc, char** argv) {
    const int dev0 = argc > 1 ? atoi(argv[1]) : 0;
    const int dev1 = argc > 2 ? atoi(argv[2]) : 1;
    const size_t bytes = 21ull << 20;

    CHECK(cudaSetDevice(dev0)); CHECK(cudaFree(0));
    CHECK(cudaSetDevice(dev1)); CHECK(cudaFree(0));
    CHECK(cudaSetDevice(dev0)); CHECK(cudaDeviceEnablePeerAccess(dev1, 0));
    CHECK(cudaSetDevice(dev1)); CHECK(cudaDeviceEnablePeerAccess(dev0, 0));
    CHECK(cudaSetDevice(dev0));
    void* a = nullptr; CHECK(cudaMalloc(&a, bytes)); CHECK(cudaMemset(a, 1, bytes));
    CHECK(cudaSetDevice(dev1));
    void* b = nullptr; CHECK(cudaMalloc(&b, bytes)); CHECK(cudaMemset(b, 2, bytes));

    cudaEvent_t ev0, ev1;
    CHECK(cudaSetDevice(dev0));
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    for (int mode = 0; mode < 2; ++mode) {
        cudaStream_t s = nullptr;
        if (mode == 1) { CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
        for (int warm = 0; warm < 3; ++warm) {
            CHECK(cudaMemcpyPeerAsync(b, dev1, a, dev0, bytes, s));
        }
        CHECK(cudaSetDevice(dev0)); CHECK(cudaDeviceSynchronize());
        CHECK(cudaSetDevice(dev1)); CHECK(cudaDeviceSynchronize());
        CHECK(cudaSetDevice(dev0));
        const int iters = 20;
        CHECK(cudaEventRecord(ev0, s));
        for (int i = 0; i < iters; ++i) {
            CHECK(cudaMemcpyPeerAsync(b, dev1, a, dev0, bytes, s));
        }
        CHECK(cudaEventRecord(ev1, s));
        CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        printf("memcpyPeerAsync %s stream: %.3f ms  => %.2f GB/s\n",
               mode == 0 ? "NULL" : "nonblocking", ms / iters,
               static_cast<double>(bytes) / 1e9 / (ms / iters / 1e3));
        if (mode == 1) { CHECK(cudaStreamDestroy(s)); }
    }

    // Concurrent both-direction copies on two devices, non-blocking streams.
    {
        cudaStream_t s0 = nullptr, s1 = nullptr;
        CHECK(cudaSetDevice(dev0)); CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
        CHECK(cudaSetDevice(dev1)); CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        for (int warm = 0; warm < 3; ++warm) {
            CHECK(cudaSetDevice(dev0)); CHECK(cudaMemcpyPeerAsync(b, dev1, a, dev0, bytes, s0));
            CHECK(cudaSetDevice(dev1)); CHECK(cudaMemcpyPeerAsync(a, dev0, b, dev1, bytes, s1));
        }
        CHECK(cudaSetDevice(dev0)); CHECK(cudaDeviceSynchronize());
        CHECK(cudaSetDevice(dev1)); CHECK(cudaDeviceSynchronize());
        CHECK(cudaSetDevice(dev0));
        const int iters = 20;
        CHECK(cudaEventRecord(ev0, s0));
        for (int i = 0; i < iters; ++i) {
            CHECK(cudaSetDevice(dev0)); CHECK(cudaMemcpyPeerAsync(b, dev1, a, dev0, bytes, s0));
            CHECK(cudaSetDevice(dev1)); CHECK(cudaMemcpyPeerAsync(a, dev0, b, dev1, bytes, s1));
        }
        CHECK(cudaSetDevice(dev0)); CHECK(cudaEventRecord(ev1, s0));
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaSetDevice(dev1)); CHECK(cudaStreamSynchronize(s1));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        printf("concurrent both-dir on nonblocking streams: %.3f ms per round => %.2f GB/s per dir\n",
               ms / iters, static_cast<double>(bytes) / 1e9 / (ms / iters / 1e3));
    }

    printf("MEMCPY DEBUG DONE\n");
    return 0;
}
