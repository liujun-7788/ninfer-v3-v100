// NCCL allreduce bench: 2 ranks in one process, decode and prefill sizes, eager + graph.
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(x)                                                                \
    do {                                                                        \
        cudaError_t e = (x);                                                    \
        if (e != cudaSuccess) {                                                 \
            printf("ERR %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define NCCLCHECK(x)                                                            \
    do {                                                                        \
        ncclResult_t r = (x);                                                   \
        if (r != ncclSuccess) {                                                 \
            printf("NCCL_ERR %s @%d: %s\n", #x, __LINE__, ncclGetErrorString(r)); \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main(int argc, char** argv) {
    const int dev0 = argc > 1 ? atoi(argv[1]) : 0;
    const int dev1 = argc > 2 ? atoi(argv[2]) : 1;

    ncclComm_t comms[2];
    int devs[2] = {dev0, dev1};
    NCCLCHECK(ncclCommInitAll(comms, 2, devs));

    void* buf[2];
    for (int r = 0; r < 2; ++r) {
        CHECK(cudaSetDevice(devs[r]));
        CHECK(cudaMalloc(&buf[r], 64u << 20));
        CHECK(cudaMemset(buf[r], 1, 64u << 20));
    }

    cudaEvent_t ev0, ev1;
    CHECK(cudaSetDevice(dev0));
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    for (int trial = 0; trial < 2; ++trial) {
        const long count = trial == 0 ? 5120L : 2048L * 5120L;
        cudaStream_t s0 = nullptr;
        cudaStream_t s1 = nullptr;
        CHECK(cudaSetDevice(dev0));
        CHECK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
        CHECK(cudaSetDevice(dev1));
        CHECK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

        // warmup eager
        for (int i = 0; i < 10; ++i) {
            CHECK(cudaSetDevice(dev0));
            NCCLCHECK(ncclAllReduce(buf[0], buf[0], count, ncclBfloat16, ncclSum, comms[0], s0));
            CHECK(cudaSetDevice(dev1));
            NCCLCHECK(ncclAllReduce(buf[1], buf[1], count, ncclBfloat16, ncclSum, comms[1], s1));
        }
        CHECK(cudaSetDevice(dev0)); CHECK(cudaStreamSynchronize(s0));
        CHECK(cudaSetDevice(dev1)); CHECK(cudaStreamSynchronize(s1));

        // eager timed
        const int iters = trial == 0 ? 200 : 30;
        CHECK(cudaSetDevice(dev0));
        CHECK(cudaEventRecord(ev0, s0));
        for (int i = 0; i < iters; ++i) {
            CHECK(cudaSetDevice(dev0));
            NCCLCHECK(ncclAllReduce(buf[0], buf[0], count, ncclBfloat16, ncclSum, comms[0], s0));
            CHECK(cudaSetDevice(dev1));
            NCCLCHECK(ncclAllReduce(buf[1], buf[1], count, ncclBfloat16, ncclSum, comms[1], s1));
        }
        CHECK(cudaEventRecord(ev1, s0));
        CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        printf("nccl %s eager: %.1f us/call", trial == 0 ? "5120" : "21MB", ms / iters * 1e3f);
        if (trial == 1) { printf("  (%.2f GB/s alg)", 21.47 / (ms / iters)); }
        printf("\n");

        // graph capture timed
        cudaGraph_t g0 = nullptr, g1 = nullptr;
        CHECK(cudaSetDevice(dev0));
        CHECK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeThreadLocal));
        NCCLCHECK(ncclAllReduce(buf[0], buf[0], count, ncclBfloat16, ncclSum, comms[0], s0));
        CHECK(cudaStreamEndCapture(s0, &g0));
        CHECK(cudaSetDevice(dev1));
        CHECK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
        NCCLCHECK(ncclAllReduce(buf[1], buf[1], count, ncclBfloat16, ncclSum, comms[1], s1));
        CHECK(cudaStreamEndCapture(s1, &g1));
        cudaGraphExec_t e0 = nullptr, e1 = nullptr;
        CHECK(cudaGraphInstantiate(&e0, g0, nullptr, nullptr, 0));
        CHECK(cudaGraphInstantiate(&e1, g1, nullptr, nullptr, 0));
        const int giters = trial == 0 ? 100 : 30;
        CHECK(cudaSetDevice(dev0));
        CHECK(cudaEventRecord(ev0, s0));
        for (int i = 0; i < giters; ++i) {
            CHECK(cudaSetDevice(dev0));
            CHECK(cudaGraphLaunch(e0, s0));
            CHECK(cudaSetDevice(dev1));
            CHECK(cudaGraphLaunch(e1, s1));
        }
        CHECK(cudaEventRecord(ev1, s0));
        CHECK(cudaEventSynchronize(ev1));
        CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        printf("nccl %s graph: %.1f us/call\n", trial == 0 ? "5120" : "21MB", ms / giters * 1e3f);
        CHECK(cudaGraphExecDestroy(e0));
        CHECK(cudaGraphExecDestroy(e1));
        CHECK(cudaGraphDestroy(g0));
        CHECK(cudaGraphDestroy(g1));
        CHECK(cudaStreamDestroy(s0));
        CHECK(cudaStreamDestroy(s1));
    }

    printf("NCCL BENCH DONE\n");
    return 0;
}
