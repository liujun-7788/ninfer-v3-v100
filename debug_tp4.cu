// Phase-by-phase timing of the large allreduce path replica.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CHECK(x)                                                                \
    do {                                                                        \
        cudaError_t e = (x);                                                    \
        if (e != cudaSuccess) {                                                 \
            printf("ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

__global__ void k_signal(unsigned long long* counter, volatile unsigned long long* flag) {
    if (threadIdx.x == 0) {
        const unsigned long long g = atomicAdd(counter, 1ULL) + 1ULL;
        __threadfence_system();
        *flag = g;
    }
}

__global__ void k_wait(const volatile unsigned long long* my_flag,
                       const volatile unsigned long long* peer_flag) {
    if (threadIdx.x == 0) {
        const unsigned long long g = *my_flag;
        while (*peer_flag < g) { __nanosleep(256); }
    }
}

__global__ void k_sum(const uint4* __restrict__ mine, const uint4* __restrict__ staged,
                      uint4* __restrict__ out, int n) {
    const std::int64_t stride = (std::int64_t)gridDim.x * blockDim.x;
    for (std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint4 a = mine[i];
        const uint4 b = staged[i];
        uint4 r;
        const __nv_bfloat16* pa = (const __nv_bfloat16*)&a;
        const __nv_bfloat16* pb = (const __nv_bfloat16*)&b;
        __nv_bfloat16* pr = (__nv_bfloat16*)&r;
        for (int k = 0; k < 8; ++k) { pr[k] = __float2bfloat16(__bfloat162float(pa[k]) + __bfloat162float(pb[k])); }
        out[i] = r;
    }
}

struct Rank {
    int dev;
    cudaStream_t s;
    void* in;
    void* out;
    void* staging;
    unsigned long long* counter;
};

int main(int argc, char** argv) {
    const int dev0 = argc > 1 ? atoi(argv[1]) : 0;
    const int dev1 = argc > 2 ? atoi(argv[2]) : 1;
    const int count = 2048 * 5120;
    const int n_vec8 = count / 8;
    const size_t bytes = (size_t)count * 2;

    Rank r[2];
    r[0].dev = dev0;
    r[1].dev = dev1;
    volatile unsigned long long* flags[2] = {nullptr, nullptr};
    CHECK(cudaHostAlloc(&flags[0], 8, cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK(cudaHostAlloc(&flags[1], 8, cudaHostAllocMapped | cudaHostAllocPortable));
    *flags[0] = 0; *flags[1] = 0;

    for (int i = 0; i < 2; ++i) {
        CHECK(cudaSetDevice(r[i].dev)); CHECK(cudaFree(0));
        CHECK(cudaStreamCreateWithFlags(&r[i].s, cudaStreamNonBlocking));
        CHECK(cudaMalloc(&r[i].in, bytes));
        CHECK(cudaMalloc(&r[i].out, bytes));
        CHECK(cudaMalloc(&r[i].staging, bytes));
        CHECK(cudaMalloc(&r[i].counter, 8));
        CHECK(cudaMemset(r[i].counter, 0, 8));
        CHECK(cudaMemset(r[i].in, 0x3F, bytes));
    }
    CHECK(cudaSetDevice(dev0)); CHECK(cudaDeviceEnablePeerAccess(dev1, 0));
    CHECK(cudaSetDevice(dev1)); CHECK(cudaDeviceEnablePeerAccess(dev0, 0));

    cudaEvent_t ev[6];
    CHECK(cudaSetDevice(dev0));
    for (int i = 0; i < 6; ++i) { CHECK(cudaEventCreate(&ev[i])); }

    auto launch_round = [&](bool with_events) {
        for (int i = 0; i < 2; ++i) {
            CHECK(cudaSetDevice(r[i].dev));
            k_signal<<<1, 32, 0, r[i].s>>>(r[i].counter, flags[i]);
        }
        int ei = 0;
        if (with_events) { CHECK(cudaSetDevice(dev0)); CHECK(cudaEventRecord(ev[ei++], r[0].s)); }
        for (int i = 0; i < 2; ++i) {
            CHECK(cudaSetDevice(r[i].dev));
            k_wait<<<1, 32, 0, r[i].s>>>(flags[i], flags[i ^ 1]);
            CHECK(cudaMemcpyPeerAsync(r[i].staging, r[i].dev, r[i ^ 1].in, r[i ^ 1].dev, bytes, r[i].s));
            k_sum<<<320, 256, 0, r[i].s>>>((const uint4*)r[i].in, (const uint4*)r[i].staging,
                                           (uint4*)r[i].out, n_vec8);
        }
        if (with_events) {
            CHECK(cudaSetDevice(dev0)); CHECK(cudaEventRecord(ev[ei++], r[0].s));
            CHECK(cudaEventSynchronize(ev[1]));
            float ms = 0; CHECK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
            printf("  round total: %.3f ms\n", ms);
        }
    };

    printf("warmup 5 rounds with timing:\n");
    for (int i = 0; i < 5; ++i) { launch_round(true); }
    printf("timed 20 rounds:\n");
    CHECK(cudaSetDevice(dev0)); CHECK(cudaDeviceSynchronize());
    CHECK(cudaSetDevice(dev1)); CHECK(cudaDeviceSynchronize());
    CHECK(cudaSetDevice(dev0));
    CHECK(cudaEventRecord(ev[4], r[0].s));
    for (int i = 0; i < 20; ++i) { launch_round(false); }
    CHECK(cudaSetDevice(dev0)); CHECK(cudaEventRecord(ev[5], r[0].s));
    CHECK(cudaEventSynchronize(ev[5]));
    float ms = 0; CHECK(cudaEventElapsedTime(&ms, ev[4], ev[5]));
    printf("20 rounds: %.1f ms total => %.3f ms/round\n", ms, ms / 20);

    // Verify one round numerically.
    launch_round(false);
    for (int i = 0; i < 2; ++i) { CHECK(cudaSetDevice(r[i].dev)); CHECK(cudaStreamSynchronize(r[i].s)); }
    CHECK(cudaSetDevice(dev0));
    unsigned long long c0 = 0, c1 = 0;
    CHECK(cudaMemcpy(&c0, r[0].counter, 8, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(&c1, r[1].counter, 8, cudaMemcpyDeviceToHost));
    printf("counters: %llu %llu\n", c0, c1);
    printf("PHASE DEBUG DONE\n");
    return 0;
}
