#pragma once

// Fused-dequant NVFP4 x BF16 GEMM on Volta tensor cores (mma.sync.m8n8k4), sm_70 only.
//
// Wide-tile sibling of nvfp4_volta_mma_gemm.cuh (the T-tile-32 quadpair form). This kernel
// covers 128 tokens x 64 output rows per CTA with 8 warps: the raw NVFP4 weight plane is
// re-read T/128 times instead of T/32, each warp decodes its 8 rows once per K step into
// shared memory and feeds four 32-row T subtiles from it, and no FP16 weight staging ever
// touches global memory (the FP16-staging CUTLASS route writes and re-reads the whole
// weight as FP16 every call).
//
// Decode is the checkpoint-native raw addressing from nvfp4_volta_mma_gemm.cuh
// (row-major code plane, BlockScaleK16M128x4 scale swizzle), so it consumes the raw
// checkpoint layout directly -- including TP2 shards, which are kept raw.
//
// Pipeline shape is q4_volta_mma_gemm.cuh's: global loads issue into register carries,
// the MMA consumes the previous K step's shared buffers, and only then do the carries
// decode-and-store into the next buffer. Fusing load+decode+store per step exposes the
// global latency with nothing to hide (see q4's note).

#include "ops/common/volta_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Nvfp4VoltaTmmaSchedule {
    static constexpr int kWarps  = 8;   // warps per CTA; each owns 8 output rows
    static constexpr int kKStep  = 32;  // K elements staged per iteration (double-buffered)
    static constexpr int kTTile  = 128; // rows of the mma A operand (4 x 32-row subtiles)
    static constexpr int kXPad   = 8;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kRowsPerCta = kWarps * 8; // 64 output rows per CTA
    static constexpr int kSubtiles = kTTile / 32;
};

// Un-permutes nvfp4_decode_e2m1_quad's (i, i+4) output into adjacent-k half2 pairs.
__device__ __forceinline__ void nvfp4_tmma_unshuffle(const half2 (&out)[4], half2 (&adj)[4]) {
    const auto* o = reinterpret_cast<const std::uint32_t*>(out);
    auto* a       = reinterpret_cast<std::uint32_t*>(adj);
    a[0] = (o[0] & 0x0000FFFFu) | (o[1] << 16);
    a[1] = (o[2] & 0x0000FFFFu) | (o[3] << 16);
    a[2] = (o[0] >> 16) | (o[1] & 0xFFFF0000u);
    a[3] = (o[2] >> 16) | (o[3] & 0xFFFF0000u);
}

__global__ __launch_bounds__(Nvfp4VoltaTmmaSchedule::kThreads, 4) void nvfp4_volta_tmma_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int out_ld, int n,
    int k, int t, float inverse_weight_divisor) {
    using S = Nvfp4VoltaTmmaSchedule;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n0   = (static_cast<int>(blockIdx.x) * S::kWarps + warp) * 8;
    const int t0   = static_cast<int>(blockIdx.z) * S::kTTile;

    __shared__ __align__(16) __half x_sh[2][S::kTTile][S::kKStep + S::kXPad];
    __shared__ __align__(16) __half w_sh[2][S::kWarps][8][S::kKStep + S::kXPad];

    const half2 rebias   = __float2half2_rn(16384.0f);
    const half2 divisor2 = __float2half2_rn(inverse_weight_divisor * 256.0f);
    const int scale_tiles = k / 64; // BlockScaleK16M128x4 tile stride, 4 groups per tile

    struct Carry {
        uint4 xraw;
        std::uint32_t word;
        std::uint8_t scale_byte;
        bool xactive;
        bool row_good;
    };
    Carry carry;

    auto prefetch = [&](int kbase, Carry& r) {
        // Activations: 128 rows x 4 uint4 vecs = 512 threads, one vec each.
        const int idx  = static_cast<int>(threadIdx.x);
        const int xrow = idx / 4;
        const int v    = idx % 4;
        r.xactive      = t0 + xrow < t;
        if (r.xactive) {
            r.xraw = *reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(t0 + xrow) * k + kbase + v * 8);
        }
        // Weights: the warp's 8 rows x 4 lane-windows, one uint32 (8 weights) per lane.
        const int r_row = lane >> 2;
        const int q     = lane & 3;
        const int row   = n0 + r_row;
        r.row_good      = row < n;
        r.word          = 0;
        r.scale_byte    = 0;
        if (r.row_good) {
            const std::uint8_t* crow = codes + static_cast<std::int64_t>(row) * (k / 2);
            r.word = *reinterpret_cast<const std::uint32_t*>(crow + (kbase + q * 8) / 2);

            const int group         = kbase / 16 + (q >> 1);
            const int m_tile        = row / 128;
            const int row_inner     = row - m_tile * 128;
            const int scale_tile    = group / 4;
            const int scale_lane    = group & 3;
            const std::int64_t off = static_cast<std::int64_t>(m_tile) * scale_tiles * 512 +
                                     static_cast<std::int64_t>(scale_tile) * 512 +
                                     (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
            r.scale_byte = scales[off];
        }
    };

    auto commit = [&](const Carry& r, int buf) {
        const int idx = static_cast<int>(threadIdx.x);
        if (r.xactive) {
            const auto* src = reinterpret_cast<const __nv_bfloat16*>(&r.xraw);
            __half tmp[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { tmp[j] = __float2half(__bfloat162float(src[j])); }
            *reinterpret_cast<uint4*>(&x_sh[buf][idx / 4][(idx % 4) * 8]) =
                *reinterpret_cast<const uint4*>(tmp);
        }
        const int r_row = lane >> 2;
        const int q     = lane & 3;
        const half2 sc2 = __hmul2(nvfp4_decode_e4m3_scale(r.scale_byte), divisor2);
        half2 raw[4];
        nvfp4_decode_e2m1_quad(r.word, rebias, raw);
        half2 adj[4];
        nvfp4_tmma_unshuffle(raw, adj);
#pragma unroll
        for (int j = 0; j < 4; ++j) { adj[j] = __hmul2(adj[j], sc2); }
        *reinterpret_cast<uint4*>(&w_sh[buf][warp][r_row][q * 8]) =
            *reinterpret_cast<const uint4*>(adj);
    };

    float d[S::kSubtiles][8];
#pragma unroll
    for (int st = 0; st < S::kSubtiles; ++st) {
#pragma unroll
        for (int l = 0; l < 8; ++l) { d[st][l] = 0.0f; }
    }

    prefetch(0, carry);
    commit(carry, 0);
    int buf = 0;
    for (int kbase = 0; kbase < k; kbase += S::kKStep) {
        __syncthreads();
        const int nxt       = kbase + S::kKStep;
        const bool has_next = nxt < k;
        if (has_next) { prefetch(nxt, carry); }

#pragma unroll
        for (int st = 0; st < S::kSubtiles; ++st) {
#pragma unroll
            for (int kk = 0; kk < S::kKStep; kk += 8) {
                half2 a[4], b[4];
                volta_load_qp(a, reinterpret_cast<const half2*>(&x_sh[buf][st * 32][kk]),
                              (S::kKStep + S::kXPad) / 2);
                volta_load_k(b, reinterpret_cast<const half2*>(&w_sh[buf][warp][0][kk]),
                             (S::kKStep + S::kXPad) / 2);
                volta_mma_qk(d[st], a, b);
            }
        }

        if (has_next) { commit(carry, buf ^ 1); }
        buf ^= 1;
    }

#pragma unroll
    for (int st = 0; st < S::kSubtiles; ++st) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int row_t = st * 32 + volta_d_get_i(l);
            const int col_n = n0 + volta_d_get_j(l);
            if (t0 + row_t < t && col_n < n) {
                out[static_cast<std::int64_t>(t0 + row_t) * out_ld + col_n] =
                    __float2bfloat16(d[st][l]);
            }
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
