#pragma once

// Act-stationary fused-dequant NVFP4 x BF16 GEMM on Volta tensor cores (mma.sync.m8n8k4),
// sm_70 only. Path B: the fused-dequant alternative to the FP16-staging CUTLASS route
// (nvfp4_cutlass_sm70.cu), which materializes the whole weight as FP16 in global memory
// on every call and then re-reads that staging buffer once per 128-token tile, so
// per-call HBM traffic is O(weight * T/128) regardless of how small the token tile gets --
// 3.2 GB per gate_up call at TP2 widths (~12 GB single-card), which is the small-chunk
// prefill cliff (2048 -> 1170, 512 -> 960, 256 -> 707 tok/s) and most of the prefill budget.
// VERDICT (measured, see the occupancy note below): correct and traffic-optimal, but on
// Volta the mma.sync.f32 latency wall leaves it 25-35% behind the CUTLASS route's tensor
// throughput at TP2 shapes, so it ships env-gated (NINFER_NVFP4_AS) and default OFF; the
// FP16-staging route stays the production path.
//
// This kernel never stages the weight. One CTA owns a 128-token x 128-output-row tile; the
// activation tile is fetched from global memory once, staged in shared memory, and reused by
// every output row in the CTA ("act-stationary"); the packed weight streams through shared
// memory one 32-wide K step at a time, decoded on the fly with the checkpoint-native decoders.
// Per-call traffic collapses to weight*(T/128) + activations*(N/128) + out: at TP2 gate_up
// (n=17408, k=5120) that is ~0.8 GB + ~1.4 GB at T=2048 and ~0.15 GB at T=256, against the
// staging route's flat ~2.9 GB weight re-read plus staging writes.
//
// Structure:
//
//   - CTA = 8 warps x 8 output rows x 2 passes = 128 output rows. The two passes own
//     disjoint 64-row halves of the CTA's N range; BOTH are resident in shared memory per
//     K step (w_sh carries a pass dimension) and both accumulate over the FULL K range --
//     an earlier draft instead interleaved the passes across K steps (pass p taking every
//     other step), which silently halves every column's dot product: the passes split the
//     N range, not K, so each column was fed by only its pass's steps. Diagnosed with a
//     constant-input probe before anything shipped.
//
//   - Occupancy: 256 threads, 41 KB shared (x_sh 20.5 KB + w_sh 20.5 KB), 2 CTAs/SM = 16
//     warps -- and this is the design's Volta ceiling. The 64 fp32 accumulators/thread
//     (2 passes x 4 subtiles x 8) plus decode/addressing temps pin the CTA to ~120
//     registers, and every occupancy-raising variant measured (kTTile 64, kKStep 16,
//     minBlocks 3-4) spilled and collapsed to 4-19 TFLOPS. Volta's synchronous
//     mma.sync.f32 (~22-cycle latency, no per-warp overlap) therefore caps this kernel at
//     16 warps / 22 clk = ~45 TFLOPS, versus ~55-68 for the CUTLASS 128x128 config's 24
//     warps. Measured at production shapes (same GPU, cudaEvent): TP2 gate_up 8.1 ms vs
//     the staging route's ~5.5; TP2 end-to-end prefill 1.04k vs 1.20k tok/s at chunk 2048
//     and 875 vs 953 at chunk 512. The staging traffic this kernel eliminates is real but
//     smaller than the tensor-throughput deficit at TP2 weight sizes, so the route ships
//     env-gated and DEFAULT OFF on TP2 (NINFER_NVFP4_AS); it is only competitive where
//     the staging route drowns in its own FP16 re-read (single-card wide weights).
//     Accumulators are 2 x 4 x 8 = 64 fp32 registers/thread.
//
//   - Pipeline shape is q4_volta_mma_gemm.cuh's, kept because that file measured it: global
//     loads issue into a register carry, the MMA consumes the previous K step's shared
//     buffers, and only then does the carry decode-and-store into the next buffer. Fusing
//     load+decode+store per step leaves global latency exposed with nothing to hide it.
//
//   - The activation prefetch covers all 128 rows (256 threads x 2 uint4). The wide-tile
//     sibling nvfp4_volta_tmma_gemm.cuh prefetched only rows 0-63 of its 128-row tile and
//     silently zeroed the rest -- verified standalone (bad=512/1024) before this kernel was
//     written; that kernel is fixed separately, and this one is built with the full-width
//     prefetch from the start.
//
//   - Weight addressing serves BOTH checkpoint layouts, selected by the launcher: raw
//     row-major codes with the BlockScaleK16M128x4 scale swizzle, and the VoltaQpnPrepacked
//     32-row x 16-k tuple layout that prepack_qpn_kernel (nvfp4_prepack_sm70.cu) produces
//     for the gate_up shards (both TP2 and single-card) and that linear_add's prepack
//     produces for down. The tuple mapping below is the exact inverse of
//     dequant_nvfp4_qpn_to_fp16 in nvfp4_cutlass_sm70.cu -- that kernel is the spec; do
//     not re-derive it. One layout subtlety cost a debugging cycle: the prepack stores
//     each 16-k group's codes in the quad decoder's own (i, i+4) pairing order
//     (prepack_qpn_kernel's order[] = {0,2,4,6,1,3,5,7,...}), so the QPN path must NOT
//     apply the adjacent-k unshuffle -- the quad decode of an engine-prepacked word is
//     already adjacent-k. Verified three-way (host reference vs CUTLASS route vs this
//     kernel) at production shapes after the fix: cutlass and this kernel agree exactly.
//
// Decode is verbatim from nvfp4_volta_qpn_gemm.cuh (e2m1 shift+rebias, e4m3 scale with the
// 256x divisor fold), and the (i, i+4) -> adjacent-k unshuffle is the same four shift+mask
// ops the TMMA/MMA kernels use.

#include "ops/common/volta_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Nvfp4VoltaAsSchedule {
    static constexpr int kWarps      = 8;   // warps per CTA
    static constexpr int kKStep      = 32;  // K elements staged per iteration (double-buffered)
    static constexpr int kTTile      = 128; // token rows per CTA (4 x 32-row mma subtiles)
    static constexpr int kPasses     = 2;   // disjoint 64-row output halves per CTA
    static constexpr int kXPad       = 8;
    static constexpr int kThreads    = kWarps * 32;
    static constexpr int kRowsPerCta = kWarps * 8 * kPasses; // 128 output rows
    static constexpr int kSubtiles   = kTTile / 32;
};

// Un-permutes nvfp4_decode_e2m1_quad's (i, i+4) output into natural adjacent-k half2 pairs.
// Local copy: the other kernels each carry one, and including their headers here would drag
// their __global__ kernels into this translation unit.
__device__ __forceinline__ void nvfp4_as_unshuffle(const half2 (&out)[4], half2 (&adj)[4]) {
    const auto* o = reinterpret_cast<const std::uint32_t*>(out);
    auto* a       = reinterpret_cast<std::uint32_t*>(adj);
    a[0] = (o[0] & 0x0000FFFFu) | (o[1] << 16);
    a[1] = (o[2] & 0x0000FFFFu) | (o[3] << 16);
    a[2] = (o[0] >> 16) | (o[1] & 0xFFFF0000u);
    a[3] = (o[2] >> 16) | (o[3] & 0xFFFF0000u);
}

__global__ __launch_bounds__(Nvfp4VoltaAsSchedule::kThreads, 2) void nvfp4_volta_as_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int out_ld, int n,
    int k, int t, float inverse_weight_divisor, int qpn) {
    using S = Nvfp4VoltaAsSchedule;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n0   = static_cast<int>(blockIdx.x) * S::kRowsPerCta;
    const int t0   = static_cast<int>(blockIdx.z) * S::kTTile;

    __shared__ __align__(16) __half x_sh[2][S::kTTile][S::kKStep + S::kXPad];
    __shared__ __align__(16) __half w_sh[2][S::kPasses][S::kWarps][8][S::kKStep + S::kXPad];

    const half2 rebias   = __float2half2_rn(16384.0f);
    const half2 divisor2 = __float2half2_rn(inverse_weight_divisor * 256.0f);
    const int scale_tiles = k / 64; // BlockScaleK16M128x4 tile stride, 4 groups per tile
    const int groups      = k / 16;

    struct Carry {
        uint4 xraw[2];
        std::uint32_t word[S::kPasses];
        std::uint8_t scale_byte[S::kPasses];
        bool xactive[2];
        bool row_good[S::kPasses];
    };
    Carry carry;

    auto prefetch = [&](int kbase, Carry& r) {
        // Activations: 128 rows x 4 uint4 vecs = 512 vecs; 256 threads, two vecs each, so the
        // whole tile is staged every step (see the header note on the TMMA coverage bug).
        const int idx = static_cast<int>(threadIdx.x);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int unit = idx + u * S::kThreads;
            const int xrow = unit / 4;
            const int v    = unit % 4;
            r.xactive[u]   = t0 + xrow < t;
            if (r.xactive[u]) {
                r.xraw[u] = *reinterpret_cast<const uint4*>(
                    x + static_cast<std::int64_t>(t0 + xrow) * k + kbase + v * 8);
            }
        }
        // Weights: each pass's 64-row set (pass * 64 rows into the CTA's N range), the
        // warp's 8 rows of it, one 8-code word per lane (lane>>2 picks the row, lane&3 the
        // 8-k window).
        const int r_row = lane >> 2;
        const int q     = lane & 3;
#pragma unroll
        for (int p = 0; p < S::kPasses; ++p) {
            const int row = n0 + p * (S::kWarps * 8) + warp * 8 + r_row;
            r.row_good[p] = row < n;
            r.word[p]     = 0;
            r.scale_byte[p] = 0;
            if (r.row_good[p]) {
                const int group = kbase / 16 + (q >> 1);
                if (qpn) {
                    // VoltaQpnPrepacked: one tuple per (32-row block, 16-k group), 8 bytes
                    // per lane -- two 4-byte words, group_half selects the 8-k half. Inverse
                    // of dequant_nvfp4_qpn_to_fp16.
                    const int local     = row & 31;
                    const int qp        = local >> 3;
                    const int rr        = local & 7;
                    const int tuplelane = (qp << 2) | (rr & 3) | ((rr & 4) << 2);
                    const std::int64_t tuple =
                        (static_cast<std::int64_t>(row >> 5) * groups + group) * 32 + tuplelane;
                    r.word[p] =
                        *reinterpret_cast<const std::uint32_t*>(codes + tuple * 8 + (q & 1) * 4);
                    r.scale_byte[p] = scales[tuple];
                } else {
                    // Raw row-major code plane (no group padding -- kKStep=32 is a whole
                    // number of 16-k groups) with the BlockScaleK16M128x4 scale swizzle.
                    const std::uint8_t* crow = codes + static_cast<std::int64_t>(row) * (k / 2);
                    r.word[p] =
                        *reinterpret_cast<const std::uint32_t*>(crow + (kbase + q * 8) / 2);
                    const int m_tile     = row >> 7;
                    const int row_inner  = row & 127;
                    const int scale_tile = group >> 2;
                    const int scale_lane = group & 3;
                    const std::int64_t off =
                        static_cast<std::int64_t>(m_tile) * scale_tiles * 512 +
                        static_cast<std::int64_t>(scale_tile) * 512 +
                        (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
                    r.scale_byte[p] = scales[off];
                }
            }
        }
    };

    auto commit = [&](const Carry& r, int buf) {
        const int idx = static_cast<int>(threadIdx.x);
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            if (r.xactive[u]) {
                const auto* src = reinterpret_cast<const __nv_bfloat16*>(&r.xraw[u]);
                __half tmp[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) { tmp[j] = __float2half(__bfloat162float(src[j])); }
                const int unit = idx + u * S::kThreads;
                *reinterpret_cast<uint4*>(&x_sh[buf][unit / 4][(unit % 4) * 8]) =
                    *reinterpret_cast<const uint4*>(tmp);
            }
        }
        const int r_row = lane >> 2;
        const int q     = lane & 3;
#pragma unroll
        for (int p = 0; p < S::kPasses; ++p) {
            const half2 sc2 = __hmul2(nvfp4_decode_e4m3_scale(r.scale_byte[p]), divisor2);
            half2 raw[4];
            nvfp4_decode_e2m1_quad(r.word[p], rebias, raw);
            half2 adj[4];
            if (qpn) {
                // The QPN prepack stores each 16-k group's codes in the quad decoder's own
                // (i, i+4) pairing (see prepack_qpn_kernel's order[]), so the decode above
                // already yields adjacent-k half2 pairs -- unshuffling here would re-scramble
                // them. Only the raw layout needs the un-permutation.
#pragma unroll
                for (int j = 0; j < 4; ++j) { adj[j] = raw[j]; }
            } else {
                nvfp4_as_unshuffle(raw, adj);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) { adj[j] = __hmul2(adj[j], sc2); }
            *reinterpret_cast<uint4*>(&w_sh[buf][p][warp][r_row][q * 8]) =
                *reinterpret_cast<const uint4*>(adj);
        }
    };

    float d[S::kPasses][S::kSubtiles][8];
#pragma unroll
    for (int p = 0; p < S::kPasses; ++p) {
#pragma unroll
        for (int st = 0; st < S::kSubtiles; ++st) {
#pragma unroll
            for (int l = 0; l < 8; ++l) { d[p][st][l] = 0.0f; }
        }
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
#pragma unroll
                for (int p = 0; p < S::kPasses; ++p) {
                    volta_load_k(b, reinterpret_cast<const half2*>(&w_sh[buf][p][warp][0][kk]),
                                 (S::kKStep + S::kXPad) / 2);
                    volta_mma_qk(d[p][st], a, b);
                }
            }
        }

        if (has_next) { commit(carry, buf ^ 1); }
        buf ^= 1;
    }

#pragma unroll
    for (int p = 0; p < S::kPasses; ++p) {
#pragma unroll
        for (int st = 0; st < S::kSubtiles; ++st) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int row_t = st * 32 + volta_d_get_i(l);
                const int col_n = n0 + p * (S::kWarps * 8) + warp * 8 + volta_d_get_j(l);
                if (t0 + row_t < t && col_n < n) {
                    out[static_cast<std::int64_t>(t0 + row_t) * out_ld + col_n] =
                        __float2bfloat16(d[p][st][l]);
                }
            }
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
