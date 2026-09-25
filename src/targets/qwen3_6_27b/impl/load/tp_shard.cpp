#include <cstdlib>
#include <string>
#include <string_view>
#include "targets/qwen3_6_27b/impl/load/tp_shard.h"

#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/fp8/fp8_prepack_sm70.h"
#include "ops/linear/nvfp4/nvfp4_prepack_sm70.h"
#endif

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

constexpr std::uint64_t kTensorAlignment = 256;
constexpr std::size_t kStagingBytes      = 2u * 1024u * 1024u;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("tp2 weight shard: ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

void copy_d2d(void* dst, const void* src, std::uint64_t bytes) {
    if (bytes == 0) { return; }
    check_cuda(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(bytes), cudaMemcpyDeviceToDevice),
               "device-to-device copy");
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct Segment {
    std::int32_t src;   // source row (or channel) begin
    std::int32_t count; // rows (channels) in the segment
    std::int32_t dst;   // destination row begin
};

// Attention / GDN / MTP out projections and MLP down projections are K-sharded columns.
// Fused input projections, attention cores, conv states, control heads, embeddings, the
// lm_head and the MTP draft weights all stay FULL (replicated): each TP2 branch then has
// exactly two allreduce sites per layer (out_proj, MLP down), and the MTP draft runs
// redundantly on both ranks with no collectives at all.

// Fused MLP gate_up [34816, k] -> [17408, k] regrouped as gate(8704) | up(8704) per rank:
// rank r keeps gate rows [r*8704, +8704) and up rows [17408 + r*8704, +8704). All segment
// bounds are 128-row aligned so the NVFP4 per-128-row scale blocks move as whole units.
std::array<Segment, 2> gate_up_segments(int rank) {
    return {Segment{static_cast<std::int32_t>(8704 * rank), 8704, 0},
            Segment{static_cast<std::int32_t>(17408 + 8704 * rank), 8704, 8704}};
}

// Copies `rows` rows of `row_bytes` from a strided source region into a contiguous destination
// through the staging buffer. Both hops run in order on the default stream, so the staging buffer
// may be reused across batches and the destination is allowed to overlap the source.
void staged_rows_to_contiguous(std::byte* staging, std::size_t staging_bytes, void* dst,
                               const std::byte* src, std::int64_t src_pitch,
                               std::int64_t src_col_offset, std::int64_t row_bytes,
                               std::int32_t rows) {
    if (row_bytes <= 0 || rows <= 0) {
        throw std::logic_error("tp2 weight shard: empty staged copy");
    }
    std::int32_t begin = 0;
    while (begin < rows) {
        const std::int64_t batch = std::min<std::int64_t>(
            rows - begin, static_cast<std::int64_t>(staging_bytes / static_cast<std::size_t>(row_bytes)));
        check_cuda(cudaMemcpy2DAsync(staging, static_cast<std::size_t>(row_bytes),
                                     src + static_cast<std::int64_t>(begin) * src_pitch +
                                         src_col_offset,
                                     static_cast<std::size_t>(src_pitch),
                                     static_cast<std::size_t>(row_bytes),
                                     static_cast<std::size_t>(batch), cudaMemcpyDeviceToDevice),
                   "staged copy into scratch");
        check_cuda(cudaMemcpy2DAsync(static_cast<std::byte*>(dst) +
                                         static_cast<std::int64_t>(begin) * row_bytes,
                                     static_cast<std::size_t>(row_bytes), staging,
                                     static_cast<std::size_t>(row_bytes),
                                     static_cast<std::size_t>(row_bytes),
                                     static_cast<std::size_t>(batch), cudaMemcpyDeviceToDevice),
                   "staged copy out of scratch");
        begin += static_cast<std::int32_t>(batch);
    }
}

void require_shardable(const Weight& w, const char* what) {
    if (w.layout == QuantLayout::VoltaQpnPrepacked) {
        throw std::logic_error(std::string(what) +
                               ": weight is already qpn prepacked; the tp2 load path must "
                               "suppress load-time prepacks");
    }
    if (w.layout != QuantLayout::RowScale && w.layout != QuantLayout::RowSplit &&
        w.layout != QuantLayout::BlockScaleK16M128x4 && w.layout != QuantLayout::Contiguous) {
        throw std::logic_error(std::string(what) + ": unsupported quant layout for tp2 sharding");
    }
}

// Re-applies the load-time QPN prepack on the sharded layout. FP8 weights are always prepacked;
// NVFP4 weights are prepacked only for the post-mixer linears, matching the single-GPU loader.
void prepack_sharded(Weight& w, bool nvfp4_mlp) {
#ifdef NINFER_VOLTA_BUILD
    if (w.qtype == QType::FP8_E4M3FN_ROW_BF16S && w.layout == QuantLayout::RowScale) {
        ::ninfer::ops::detail::fp8_prepack_qpn_sm70(w);
    } else if (nvfp4_mlp && w.qtype == QType::NVFP4 &&
               w.layout == QuantLayout::BlockScaleK16M128x4) {
        // The wide-T routes for both MLP shards are the cutlass ones (fused swiglu for gate_up,
        // direct cutlass for down); their QPN/raw dequant kernels expect the prepacked layout.
        ::ninfer::ops::detail::nvfp4_prepack_qpn_sm70(w);
    }
#else
    (void)w;
    (void)nvfp4_mlp;
#endif
}

// Compacts the first `rows` dimension of a weight to n_new per the segment table, in place.
// Code copies run first (ascending destination, each segment destination ends before the next
// segment source begins), then the scale plane moves in a second phase, so later code sources
// are never clobbered and scale destinations never reach the old scale plane.
void shard_rows(Weight& w, int rank, std::span<const Segment> segs, std::int32_t n_new,
                const char* what) {
    require_shardable(w, what);
    static std::atomic<int> spy{0};
    if (spy.fetch_add(1) < 6) {
        std::fprintf(stderr, "TP2SPY rows %s qtype=%d layout=%d n=%d k=%d shape0=%d padded0=%d\n",
                     what, static_cast<int>(w.qtype), static_cast<int>(w.layout), w.n, w.k,
                     w.shape[0], w.padded_shape[0]);
        std::fflush(stderr);
    }
    if (w.n != 2 * n_new) {
        throw std::logic_error(std::string(what) + ": unexpected row count for tp2 sharding");
    }
    const auto* base    = static_cast<const std::byte*>(w.qdata);
    auto* dst_base      = const_cast<std::byte*>(base);
    const int tail_rank = rank; // kept symmetric with shard_columns for readability

    if (w.layout == QuantLayout::RowScale && w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        for (const Segment& s : segs) {
            copy_d2d(dst_base + static_cast<std::int64_t>(s.dst) * w.k,
                     base + static_cast<std::int64_t>(s.src) * w.k,
                     static_cast<std::uint64_t>(s.count) * w.k);
        }
        const auto* old_scales = static_cast<const std::byte*>(w.scales);
        auto* new_scales =
            dst_base + align_up(static_cast<std::uint64_t>(n_new) * w.k, kTensorAlignment);
        for (const Segment& s : segs) {
            copy_d2d(new_scales + static_cast<std::int64_t>(s.dst) * 2,
                     old_scales + static_cast<std::int64_t>(s.src) * 2,
                     static_cast<std::uint64_t>(s.count) * 2);
        }
        w.scales        = new_scales;
        w.payload_bytes = align_up(static_cast<std::uint64_t>(n_new) * w.k, kTensorAlignment) +
                          static_cast<std::uint64_t>(n_new) * 2;
        w.scale_ne[0]      = n_new;
        w.scale_nb[1]      = static_cast<std::int64_t>(n_new) * 2;
        w.scale_nb[2]      = static_cast<std::int64_t>(n_new) * 2;
        w.scale_nb[3]      = static_cast<std::int64_t>(n_new) * 2;
    } else if (w.layout == QuantLayout::RowSplit && w.qtype == QType::W8G32_F16S) {
        if (w.padded_shape[1] != w.k) {
            throw std::logic_error(std::string(what) + ": padded row split width mismatch");
        }
        const std::int64_t row_pitch =
            w.padded_shape[1]; // one byte per element of the low int8 plane
        const std::uint64_t groups = static_cast<std::uint64_t>(w.padded_shape[1] / w.group);
        for (const Segment& s : segs) {
            copy_d2d(dst_base + static_cast<std::int64_t>(s.dst) * row_pitch,
                     base + static_cast<std::int64_t>(s.src) * row_pitch,
                     static_cast<std::uint64_t>(s.count) * row_pitch);
        }
        const auto* old_scales = static_cast<const std::byte*>(w.scales);
        auto* new_scales       = dst_base + align_up(static_cast<std::uint64_t>(n_new) * row_pitch,
                                                     kTensorAlignment);
        for (const Segment& s : segs) {
            copy_d2d(new_scales + static_cast<std::int64_t>(s.dst) * groups * 2,
                     old_scales + static_cast<std::int64_t>(s.src) * groups * 2,
                     static_cast<std::uint64_t>(s.count) * groups * 2);
        }
        w.scales        = new_scales;
        w.payload_bytes = align_up(static_cast<std::uint64_t>(n_new) * row_pitch,
                                   kTensorAlignment) +
                          static_cast<std::uint64_t>(n_new) * groups * 2;
    } else if (w.layout == QuantLayout::Contiguous) {
        const std::uint64_t row_elems = static_cast<std::uint64_t>(w.k);
        const std::uint64_t elem_bytes =
            w.payload_bytes / (static_cast<std::uint64_t>(w.n) * row_elems);
        if (elem_bytes != 1 && elem_bytes != 2) {
            throw std::logic_error(std::string(what) + ": unexpected contiguous element size");
        }
        for (const Segment& s : segs) {
            copy_d2d(dst_base + static_cast<std::int64_t>(s.dst) * static_cast<std::int64_t>(row_elems) *
                                 static_cast<std::int64_t>(elem_bytes),
                     base + static_cast<std::int64_t>(s.src) * static_cast<std::int64_t>(row_elems) *
                                 static_cast<std::int64_t>(elem_bytes),
                     static_cast<std::uint64_t>(s.count) * row_elems * elem_bytes);
        }
        w.payload_bytes = static_cast<std::uint64_t>(n_new) * row_elems * elem_bytes;
    } else if (w.layout == QuantLayout::BlockScaleK16M128x4 && w.qtype == QType::NVFP4) {
        const std::int64_t code_pitch = w.k / 2; // two fp4 values per byte
        for (const Segment& s : segs) {
            copy_d2d(dst_base + static_cast<std::int64_t>(s.dst) * code_pitch,
                     base + static_cast<std::int64_t>(s.src) * code_pitch,
                     static_cast<std::uint64_t>(s.count) * code_pitch);
        }
        // Per-128-row scale blocks, each k*8 bytes (k/64 tiles of 512 bytes). Every tp2 segment
        // is 128-row aligned by construction, so scale blocks move as whole units.
        const std::uint64_t block_bytes = static_cast<std::uint64_t>(w.k) * 8;
        const auto* old_scales          = static_cast<const std::byte*>(w.scales);
        auto* new_scales =
            dst_base + align_up(static_cast<std::uint64_t>(n_new) * code_pitch, kTensorAlignment);
        for (const Segment& s : segs) {
            if (s.src % 128 != 0 || s.dst % 128 != 0 || s.count % 128 != 0) {
                throw std::logic_error(std::string(what) +
                                       ": nvfp4 tp2 segments must be 128-row aligned");
            }
            copy_d2d(new_scales + static_cast<std::int64_t>(s.dst / 128) * block_bytes,
                     old_scales + static_cast<std::int64_t>(s.src / 128) * block_bytes,
                     static_cast<std::uint64_t>(s.count) / 128 * block_bytes);
        }
        // The trailing 4-byte weight divisor sits after the scale plane and must be relocated.
        const std::uint64_t old_scale_bytes = static_cast<std::uint64_t>(w.n / 128) * block_bytes;
        const std::uint64_t new_scale_bytes = static_cast<std::uint64_t>(n_new / 128) * block_bytes;
        copy_d2d(new_scales + new_scale_bytes, old_scales + old_scale_bytes, 4);
        w.scales        = new_scales;
        w.payload_bytes = align_up(static_cast<std::uint64_t>(n_new) * code_pitch,
                                   kTensorAlignment) +
                          new_scale_bytes + 4;
    } else {
        throw std::logic_error(std::string(what) + ": unsupported weight format for tp2 rows");
    }

    (void)tail_rank;
    w.n            = n_new;
    w.shape[0]     = n_new;
    w.padded_shape[0] = n_new;
}

// Compacts the second dimension of a weight to k_new, keeping the rank-specific column half.
// Overlapping in-place column removal is staged through a scratch buffer.
void shard_columns(Weight& w, int rank, std::int32_t k_new, std::byte* staging,
                   std::size_t staging_bytes, const char* what) {
    require_shardable(w, what);
    static std::atomic<int> spy_c{0};
    if (spy_c.fetch_add(1) < 6) {
        std::fprintf(stderr, "TP2SPY cols %s qtype=%d layout=%d n=%d k=%d shape0=%d padded1=%d\n",
                     what, static_cast<int>(w.qtype), static_cast<int>(w.layout), w.n, w.k,
                     w.shape[0], w.padded_shape[1]);
        std::fflush(stderr);
    }
    if (w.k != 2 * k_new) {
        throw std::logic_error(std::string(what) + ": unexpected column count for tp2 sharding");
    }
    const auto* base = static_cast<const std::byte*>(w.qdata);
    auto* dst_base   = const_cast<std::byte*>(base);
    const std::int64_t src_col = static_cast<std::int64_t>(rank) * k_new;

    if (w.layout == QuantLayout::RowScale && w.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        staged_rows_to_contiguous(staging, staging_bytes, dst_base, base, w.k, src_col, k_new, w.n);
        const auto* old_scales = static_cast<const std::byte*>(w.scales);
        auto* new_scales =
            dst_base + align_up(static_cast<std::uint64_t>(w.n) * k_new, kTensorAlignment);
        copy_d2d(new_scales, old_scales, static_cast<std::uint64_t>(w.n) * 2);
        w.scales        = new_scales;
        w.payload_bytes = align_up(static_cast<std::uint64_t>(w.n) * k_new, kTensorAlignment) +
                          static_cast<std::uint64_t>(w.n) * 2;
        w.k             = k_new;
        w.shape[1]      = k_new;
        w.padded_shape[1] = k_new;
        w.group_size    = static_cast<std::uint32_t>(k_new);
        w.group         = k_new;
    } else if (w.layout == QuantLayout::Contiguous) {
        const std::uint64_t elem_bytes =
            w.payload_bytes / (static_cast<std::uint64_t>(w.n) * static_cast<std::uint64_t>(w.k));
        if (elem_bytes != 1 && elem_bytes != 2) {
            throw std::logic_error(std::string(what) + ": unexpected contiguous element size");
        }
        staged_rows_to_contiguous(staging, staging_bytes, dst_base, base,
                                  static_cast<std::int64_t>(w.k) *
                                      static_cast<std::int64_t>(elem_bytes),
                                  src_col * static_cast<std::int64_t>(elem_bytes),
                                  static_cast<std::int64_t>(k_new) *
                                      static_cast<std::int64_t>(elem_bytes),
                                  w.n);
        w.payload_bytes = static_cast<std::uint64_t>(w.n) * static_cast<std::uint64_t>(k_new) *
                          elem_bytes;
        w.k             = k_new;
        w.shape[1]      = k_new;
        w.padded_shape[1] = k_new;
    } else if (w.layout == QuantLayout::RowSplit && w.qtype == QType::W8G32_F16S) {
        if (k_new % w.group != 0 || w.padded_shape[1] != w.k) {
            throw std::logic_error(std::string(what) + ": row split width is not group aligned");
        }
        staged_rows_to_contiguous(staging, staging_bytes, dst_base, base, w.padded_shape[1],
                                  src_col, k_new, w.n);
        const std::uint64_t groups_old =
            static_cast<std::uint64_t>(w.padded_shape[1] / w.group);
        const std::uint64_t groups_new = static_cast<std::uint64_t>(k_new / w.group);
        auto* new_scales =
            dst_base + align_up(static_cast<std::uint64_t>(w.n) * k_new, kTensorAlignment);
        staged_rows_to_contiguous(staging, staging_bytes, new_scales,
                                  static_cast<const std::byte*>(w.scales),
                                  static_cast<std::int64_t>(groups_old) * 2,
                                  static_cast<std::int64_t>(rank * groups_new) * 2,
                                  static_cast<std::int64_t>(groups_new) * 2, w.n);
        w.scales        = new_scales;
        w.payload_bytes = align_up(static_cast<std::uint64_t>(w.n) * k_new, kTensorAlignment) +
                          static_cast<std::uint64_t>(w.n) * groups_new * 2;
        w.k             = k_new;
        w.shape[1]      = k_new;
        w.padded_shape[1] = k_new;
    } else if (w.layout == QuantLayout::BlockScaleK16M128x4 && w.qtype == QType::NVFP4) {
        if (k_new % 128 != 0 || w.k % 128 != 0) {
            throw std::logic_error(std::string(what) + ": nvfp4 tp2 widths must be 128 aligned");
        }
        staged_rows_to_contiguous(staging, staging_bytes, dst_base, base, w.k / 2,
                                  src_col / 2, k_new / 2, w.n);
        const std::uint64_t block_bytes_new = static_cast<std::uint64_t>(k_new) * 8;
        const std::uint64_t block_bytes_old = static_cast<std::uint64_t>(w.k) * 8;
        const auto* old_scales              = static_cast<const std::byte*>(w.scales);
        auto* new_scales =
            dst_base + align_up(static_cast<std::uint64_t>(w.n) * (k_new / 2), kTensorAlignment);
        const std::int32_t blocks = w.n / 128;
        for (std::int32_t b = 0; b < blocks; ++b) {
            copy_d2d(new_scales + static_cast<std::int64_t>(b) * block_bytes_new,
                     old_scales + static_cast<std::int64_t>(b) * block_bytes_old +
                         static_cast<std::int64_t>(rank) * block_bytes_new,
                     block_bytes_new);
        }
        copy_d2d(new_scales + static_cast<std::int64_t>(blocks) * block_bytes_new,
                 old_scales + static_cast<std::int64_t>(blocks) * block_bytes_old, 4);
        w.scales        = new_scales;
        w.payload_bytes = align_up(static_cast<std::uint64_t>(w.n) * (k_new / 2),
                                   kTensorAlignment) +
                          static_cast<std::uint64_t>(blocks) * block_bytes_new + 4;
        w.k             = k_new;
        w.shape[1]      = k_new;
        w.padded_shape[1] = k_new;
    } else {
        throw std::logic_error(std::string(what) + ": unsupported weight format for tp2 columns");
    }
}

} // namespace

// Halves a 1-D elementwise tensor (norm weights) in place for TP2 head-split ranks.
void shard_elementwise_half(Tensor& t, int rank, const char* what) {
    const std::size_t esize = t.numel() ? (t.bytes() / t.numel()) : 0;
    if (esize == 0 || t.numel() % 2 != 0) {
        throw std::logic_error(std::string(what) + ": cannot halve the norm tensor");
    }
    const std::size_t half = t.numel() / 2;
    const std::size_t bytes = half * esize;
    std::uint8_t* base = static_cast<std::uint8_t*>(t.data);
    if (rank == 1) {
        check_cuda(cudaMemcpy(base, base + bytes, bytes, cudaMemcpyDeviceToDevice),
                   "norm half copy");
    }
    t.ne[0] = static_cast<std::int32_t>(half);
    t.ne[1] = 1;
    t.ne[2] = 1;
    t.ne[3] = 1;
}

void tp_shard_model(RuntimeModelView& runtime, int rank) {
    if (rank != 0 && rank != 1) {
        throw std::invalid_argument("tp2 shard rank must be 0 or 1");
    }
    void* staging = nullptr;
    check_cuda(cudaMalloc(&staging, kStagingBytes), "staging allocation");
    struct StagingGuard {
        void* p;
        ~StagingGuard() { cudaFree(p); }
    } staging_guard{staging};
    auto* scratch = static_cast<std::byte*>(staging);

    // Strategy B+ sharding: only the two per-layer allreduce producers are sharded —
    // attention/GDN out projections (K-split to 3072), MLP gate_up (N-split to 17408,
    // regrouped gate|up per rank) and MLP down (K-split to 8704). Fused input projections,
    // attention/GDN cores, conv states, control heads, embeddings, the lm_head and all MTP
    // draft weights stay FULL (replicated) on both ranks.
    // Bisect switch: NINFER_TP2_SHARD=all (default) | none | outproj | mlp.
    // "none" leaves every weight full so both ranks run identical replicated compute and
    // none of the TP2 allreduce branches fire; "outproj"/"mlp" isolate one producer site.
    const char* shard_env = std::getenv("NINFER_TP2_SHARD");
    const std::string_view shard_mode =
        shard_env == nullptr ? std::string_view("all") : std::string_view(shard_env);
    const bool shard_out = shard_mode == "all" || shard_mode == "outproj";
    const bool shard_mlp = shard_mode == "all" || shard_mode == "mlp";
    const bool shard_gu   = shard_mode == "gu";
    const bool shard_down = shard_mode == "down";
    const bool shard_heads = shard_mode == "heads";
    if (shard_heads) {
        // Attention head-split (weights only; the head-offset forward is a separate change).
        // Fused QKV [14336] = q[0:6144] | key[6144:7168] | gate[7168:13312] | value[13312:14336]
        // (row order per the decode kernel's store ranges); rank r keeps the r-th half of every
        // segment -> 7168 rows. Segments are 512-row aligned, so NVFP4 scale blocks (128-row
        // granularity) move as whole units.
        for (FullAttentionWeights& layer : runtime.full_layers) {
            Weight& qkv = std::get<FusedAttentionProjectionPayload>(layer.projection)
                              .query_key_gate_value;
            const std::int64_t q_off  = rank * 3072;
            const std::int64_t kv_off = rank * 512;
            const std::vector<Segment> qkv_segs = {
                Segment{static_cast<std::int32_t>(q_off), 3072, 0},
                Segment{static_cast<std::int32_t>(6144 + kv_off), 512, 3072},
                Segment{static_cast<std::int32_t>(7168 + q_off), 3072, 3584},
                Segment{static_cast<std::int32_t>(13312 + kv_off), 512, 6656},
            };
            shard_rows(qkv, rank, qkv_segs, 7168, "attention qkv head-split");
            // query_norm/key_norm are per-head-dim weights (head_dim elements, shared across
            // heads) -- they do NOT scale with head count and must stay full.
            shard_columns(layer.output, rank, 3072, scratch, kStagingBytes,
                          "attention output");
            prepack_sharded(layer.output, false);
        }
        return;
    }
    if (shard_out || shard_mlp || shard_gu || shard_down) {
        for (FullAttentionWeights& layer : runtime.full_layers) {
            if (shard_out) {
                shard_columns(layer.output, rank, 3072, scratch, kStagingBytes, "attention output");
                prepack_sharded(layer.output, false);
            }
            if (shard_mlp || shard_gu) {
                static bool raw_done[2] = {false, false};
                static bool shard_done[2] = {false, false};
                const Weight& gw = layer.post_mixer.gate_up;
                if (!raw_done[rank]) {
                    raw_done[rank] = true;
                    std::vector<char> gbuf(gw.payload_bytes);
                    check_cuda(cudaMemcpy(gbuf.data(), gw.qdata, gw.payload_bytes,
                                          cudaMemcpyDeviceToHost), "spy d2h");
                    char gpath[64];
                    std::snprintf(gpath, sizeof(gpath), "/tmp/spy_gu_raw_r%d.bin", rank);
                    if (FILE* gf = std::fopen(gpath, "wb")) {
                        std::fwrite(gbuf.data(), 1, gbuf.size(), gf);
                        std::fclose(gf);
                    }
                    std::fprintf(stderr, "TP2SPYDUMP raw r%d bytes=%llu\n", rank,
                                 static_cast<unsigned long long>(gw.payload_bytes));
                    std::fflush(stderr);
                }
                shard_rows(layer.post_mixer.gate_up, rank, gate_up_segments(rank), 17408,
                           "mlp gate_up");
                if (!shard_done[rank]) {
                    shard_done[rank] = true;
                    std::vector<char> sbuf(gw.payload_bytes);
                    check_cuda(cudaMemcpy(sbuf.data(), gw.qdata, gw.payload_bytes,
                                          cudaMemcpyDeviceToHost), "spy d2h2");
                    char spath[64];
                    std::snprintf(spath, sizeof(spath), "/tmp/spy_gu_shard_r%d.bin", rank);
                    if (FILE* sf = std::fopen(spath, "wb")) {
                        std::fwrite(sbuf.data(), 1, sbuf.size(), sf);
                        std::fclose(sf);
                    }
                    std::fprintf(stderr, "TP2SPYDUMP shard r%d bytes=%llu\n", rank,
                                 static_cast<unsigned long long>(gw.payload_bytes));
                    std::fflush(stderr);
                }
                prepack_sharded(layer.post_mixer.gate_up, true);
            }
            if (shard_mlp || shard_down) {
                static bool draw_done[2] = {false, false};
                static bool dshard_done[2] = {false, false};
                const Weight& dw = layer.post_mixer.down;
                if (!draw_done[rank]) {
                    draw_done[rank] = true;
                    std::vector<char> gbuf(dw.payload_bytes);
                    check_cuda(cudaMemcpy(gbuf.data(), dw.qdata, dw.payload_bytes,
                                          cudaMemcpyDeviceToHost), "spy d2h");
                    char gpath[64];
                    std::snprintf(gpath, sizeof(gpath), "/tmp/spy_dn_raw_r%d.bin", rank);
                    if (FILE* gf = std::fopen(gpath, "wb")) {
                        std::fwrite(gbuf.data(), 1, gbuf.size(), gf);
                        std::fclose(gf);
                    }
                    std::fprintf(stderr, "TP2SPYDUMP dnraw r%d bytes=%llu\n", rank,
                                 static_cast<unsigned long long>(dw.payload_bytes));
                    std::fflush(stderr);
                }
                shard_columns(layer.post_mixer.down, rank, 8704, scratch, kStagingBytes, "mlp down");
                if (!dshard_done[rank]) {
                    dshard_done[rank] = true;
                    std::vector<char> sbuf(dw.payload_bytes);
                    check_cuda(cudaMemcpy(sbuf.data(), dw.qdata, dw.payload_bytes,
                                          cudaMemcpyDeviceToHost), "spy d2h2");
                    char spath[64];
                    std::snprintf(spath, sizeof(spath), "/tmp/spy_dn_shard_r%d.bin", rank);
                    if (FILE* sf = std::fopen(spath, "wb")) {
                        std::fwrite(sbuf.data(), 1, sbuf.size(), sf);
                        std::fclose(sf);
                    }
                    std::fprintf(stderr, "TP2SPYDUMP dnshard r%d bytes=%llu\n", rank,
                                 static_cast<unsigned long long>(dw.payload_bytes));
                    std::fflush(stderr);
                }
                prepack_sharded(layer.post_mixer.down, false);
            }
        }
        for (GdnWeights& layer : runtime.gdn_layers) {
            if (shard_out) {
                shard_columns(layer.output, rank, 3072, scratch, kStagingBytes, "gdn output");
                prepack_sharded(layer.output, false);
            }
            if (shard_mlp || shard_gu) {
                static bool raw_done[2] = {false, false};
                static bool shard_done[2] = {false, false};
                const Weight& gw = layer.post_mixer.gate_up;
                if (!raw_done[rank]) {
                    raw_done[rank] = true;
                    std::vector<char> gbuf(gw.payload_bytes);
                    check_cuda(cudaMemcpy(gbuf.data(), gw.qdata, gw.payload_bytes,
                                          cudaMemcpyDeviceToHost), "spy d2h");
                    char gpath[64];
                    std::snprintf(gpath, sizeof(gpath), "/tmp/spy_gu_raw_r%d.bin", rank);
                    if (FILE* gf = std::fopen(gpath, "wb")) {
                        std::fwrite(gbuf.data(), 1, gbuf.size(), gf);
                        std::fclose(gf);
                    }
                    std::fprintf(stderr, "TP2SPYDUMP raw r%d bytes=%llu\n", rank,
                                 static_cast<unsigned long long>(gw.payload_bytes));
                    std::fflush(stderr);
                }
                shard_rows(layer.post_mixer.gate_up, rank, gate_up_segments(rank), 17408,
                           "mlp gate_up");
                if (!shard_done[rank]) {
                    shard_done[rank] = true;
                    std::vector<char> sbuf(gw.payload_bytes);
                    check_cuda(cudaMemcpy(sbuf.data(), gw.qdata, gw.payload_bytes,
                                          cudaMemcpyDeviceToHost), "spy d2h2");
                    char spath[64];
                    std::snprintf(spath, sizeof(spath), "/tmp/spy_gu_shard_r%d.bin", rank);
                    if (FILE* sf = std::fopen(spath, "wb")) {
                        std::fwrite(sbuf.data(), 1, sbuf.size(), sf);
                        std::fclose(sf);
                    }
                    std::fprintf(stderr, "TP2SPYDUMP shard r%d bytes=%llu\n", rank,
                                 static_cast<unsigned long long>(gw.payload_bytes));
                    std::fflush(stderr);
                }
                prepack_sharded(layer.post_mixer.gate_up, true);
            }
            if (shard_mlp || shard_down) {
                static bool draw_done[2] = {false, false};
                static bool dshard_done[2] = {false, false};
                const Weight& dw = layer.post_mixer.down;
                if (!draw_done[rank]) {
                    draw_done[rank] = true;
                    std::vector<char> gbuf(dw.payload_bytes);
                    check_cuda(cudaMemcpy(gbuf.data(), dw.qdata, dw.payload_bytes,
                                          cudaMemcpyDeviceToHost), "spy d2h");
                    char gpath[64];
                    std::snprintf(gpath, sizeof(gpath), "/tmp/spy_dn_raw_r%d.bin", rank);
                    if (FILE* gf = std::fopen(gpath, "wb")) {
                        std::fwrite(gbuf.data(), 1, gbuf.size(), gf);
                        std::fclose(gf);
                    }
                    std::fprintf(stderr, "TP2SPYDUMP dnraw r%d bytes=%llu\n", rank,
                                 static_cast<unsigned long long>(dw.payload_bytes));
                    std::fflush(stderr);
                }
                shard_columns(layer.post_mixer.down, rank, 8704, scratch, kStagingBytes, "mlp down");
                if (!dshard_done[rank]) {
                    dshard_done[rank] = true;
                    std::vector<char> sbuf(dw.payload_bytes);
                    check_cuda(cudaMemcpy(sbuf.data(), dw.qdata, dw.payload_bytes,
                                          cudaMemcpyDeviceToHost), "spy d2h2");
                    char spath[64];
                    std::snprintf(spath, sizeof(spath), "/tmp/spy_dn_shard_r%d.bin", rank);
                    if (FILE* sf = std::fopen(spath, "wb")) {
                        std::fwrite(sbuf.data(), 1, sbuf.size(), sf);
                        std::fclose(sf);
                    }
                    std::fprintf(stderr, "TP2SPYDUMP dnshard r%d bytes=%llu\n", rank,
                                 static_cast<unsigned long long>(dw.payload_bytes));
                    std::fflush(stderr);
                }
                prepack_sharded(layer.post_mixer.down, false);
            }
        }
    }

    // MTP weights intentionally left FULL: the draft runs redundantly on both ranks with no
    // allreduces; only the head rank's egress is consumed.

    check_cuda(cudaDeviceSynchronize(), "shard synchronize");
}

} // namespace ninfer::targets::qwen3_6_27b::detail
