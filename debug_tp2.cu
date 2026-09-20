// Exact replica of the tp_allreduce_oneshot handshake, minimal.
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

__global__ void k_handshake(unsigned long long* my_counter, unsigned long long* my_flag,
                            unsigned long long* peer_flag, int site) {
    if (threadIdx.x == 0) {
        const unsigned long long generation = atomicAdd(my_counter + site, 1ULL) + 1ULL;
        __threadfence_system();
        *reinterpret_cast<volatile unsigned long long*>(my_flag + site) = generation;
        const auto* watch = reinterpret_cast<const volatile unsigned long long*>(peer_flag + site);
        while (*watch < generation) { __nanosleep(256); }
    }
    __syncthreads();
}

int main(int argc, char** argv) {
    const int dev0 = argc > 1 ? atoi(argv[1]) : 0;
    const int dev1 = argc > 2 ? atoi(argv[2]) : 1;

    // Mailbox identical to TpGroup::Mailbox: host mapped, shared by both ranks.
    struct Mailbox {
        unsigned long long counter[2][256];
        unsigned long long flag[2][256];
    };
    Mailbox* mb = nullptr;
    CHECK(cudaHostAlloc(&mb, sizeof(Mailbox), cudaHostAllocMapped | cudaHostAllocPortable));
    memset(mb, 0, sizeof(Mailbox));

    CHECK(cudaSetDevice(dev0)); CHECK(cudaFree(0));
    CHECK(cudaSetDevice(dev1)); CHECK(cudaFree(0));
    CHECK(cudaSetDevice(dev0)); CHECK(cudaDeviceEnablePeerAccess(dev1, 0));
    CHECK(cudaSetDevice(dev1)); CHECK(cudaDeviceEnablePeerAccess(dev0, 0));

    for (int round = 0; round < 3; ++round) {
        printf("round %d: launch both handshakes (320 blocks)\n", round); fflush(stdout);
        CHECK(cudaSetDevice(dev0));
        k_handshake<<<320, 256>>>(mb->counter[0], mb->flag[0], mb->flag[1], 0);
        CHECK(cudaGetLastError());
        CHECK(cudaSetDevice(dev1));
        k_handshake<<<320, 256>>>(mb->counter[1], mb->flag[1], mb->flag[0], 0);
        CHECK(cudaGetLastError());
        CHECK(cudaSetDevice(dev0));
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaSetDevice(dev1));
        CHECK(cudaDeviceSynchronize());
        printf("  round %d OK: counter0=%llu counter1=%llu flag0=%llu flag1=%llu\n", round,
               (unsigned long long)mb->counter[0][0], (unsigned long long)mb->counter[1][0],
               (unsigned long long)mb->flag[0][0], (unsigned long long)mb->flag[1][0]);
        fflush(stdout);
    }
    printf("HANDSHAKE DONE\n");
    return 0;
}
