#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(x)                                                              \
    do {                                                                      \
        cudaError_t e = (x);                                                  \
        if (e != cudaSuccess) {                                               \
            printf("  CUDA_ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 3) { printf("usage: %s devA devB [bytes]\n", argv[0]); return 2; }
    const int a = atoi(argv[1]);
    const int b = atoi(argv[2]);
    const size_t bytes = (argc > 3) ? static_cast<size_t>(atoll(argv[3])) : (64ull << 20);

    int can_ab = 0, can_ba = 0;
    CHECK(cudaDeviceCanAccessPeer(&can_ab, a, b));
    CHECK(cudaDeviceCanAccessPeer(&can_ba, b, a));
    printf("pair(%d,%d) canAccessPeer a->b=%d b->a=%d\n", a, b, can_ab, can_ba);

    CHECK(cudaSetDevice(a)); CHECK(cudaFree(0));
    CHECK(cudaSetDevice(b)); CHECK(cudaFree(0));
    if (can_ab) { CHECK(cudaSetDevice(a)); CHECK(cudaDeviceEnablePeerAccess(b, 0)); }
    if (can_ba) { CHECK(cudaSetDevice(b)); CHECK(cudaDeviceEnablePeerAccess(a, 0)); }

    CHECK(cudaSetDevice(a));
    float* buf_a = nullptr; CHECK(cudaMalloc(&buf_a, bytes)); CHECK(cudaMemset(buf_a, 1, bytes));
    CHECK(cudaSetDevice(b));
    float* buf_b = nullptr; CHECK(cudaMalloc(&buf_b, bytes)); CHECK(cudaMemset(buf_b, 2, bytes));

    cudaEvent_t ev0, ev1;
    CHECK(cudaSetDevice(a));
    CHECK(cudaEventCreate(&ev0)); CHECK(cudaEventCreate(&ev1));

    // Bandwidth: a->b then b->a, large message. All enqueues/events stay on device a's stream;
    // cross-device peer copies issued from a's context are legal for both directions.
    CHECK(cudaSetDevice(a));
    for (int dir = 0; dir < 2; ++dir) {
        const int src = dir == 0 ? a : b;
        const int dst = dir == 0 ? b : a;
        float* src_p = dir == 0 ? buf_a : buf_b;
        float* dst_p = dir == 0 ? buf_b : buf_a;
        for (int i = 0; i < 3; ++i) CHECK(cudaMemcpyPeerAsync(dst_p, dst, src_p, src, bytes, 0));
        CHECK(cudaDeviceSynchronize());
        const int iters = 10;
        CHECK(cudaEventRecord(ev0, 0));
        for (int i = 0; i < iters; ++i)
            CHECK(cudaMemcpyPeerAsync(dst_p, dst, src_p, src, bytes, 0));
        CHECK(cudaEventRecord(ev1, 0));
        CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        printf("  bw %d->%d %zu B: %.3f ms avg  => %.2f GB/s\n", src, dst, bytes, ms / iters,
               static_cast<double>(bytes) / 1e9 / (ms / iters / 1e3));
    }

    // Latency: 16 KB messages, a->b
    {
        const size_t small = 16 * 1024;
        const int iters = 1000;
        for (int i = 0; i < 20; ++i) CHECK(cudaMemcpyPeerAsync(buf_b, b, buf_a, a, small, 0));
        CHECK(cudaSetDevice(a)); CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(ev0, 0));
        for (int i = 0; i < iters; ++i) CHECK(cudaMemcpyPeerAsync(buf_b, b, buf_a, a, small, 0));
        CHECK(cudaEventRecord(ev1, 0));
        CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f; CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        printf("  latency 16KB a->b: %.1f us avg\n", ms / iters * 1e3);
    }

    CHECK(cudaSetDevice(a)); CHECK(cudaFree(buf_a));
    CHECK(cudaSetDevice(b)); CHECK(cudaFree(buf_b));
    printf("pair(%d,%d) done\n", a, b);
    return 0;
}
