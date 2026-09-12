#include "ninfer/ops/linear_topk.h"

#include "core/layout.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear_topk/dflash2_linear_topk_volta.h"
#include "ops/linear_topk/linear_topk_launch.h"
#include "ninfer/ops/linear.h"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

enum class HeadProfile : std::uint8_t {
    W8Full,
    Fp8Full,
    Q4Optimized,
};

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

bool overlaps(const void* lhs, std::size_t lhs_bytes, const void* rhs, std::size_t rhs_bytes) {
    if (lhs == nullptr || rhs == nullptr || lhs_bytes == 0 || rhs_bytes == 0) { return false; }
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
    return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    return overlaps(lhs.data, lhs.bytes(), rhs.data, rhs.bytes());
}

HeadProfile resolve_profile(QType qtype, std::int32_t head_rows, std::int32_t input_rows) {
    if (input_rows != detail::kLinearTopKHidden) {
        throw std::invalid_argument("linear_topk: unsupported head profile");
    }
    if (head_rows == detail::kLinearTopKFullRows && qtype == QType::W8G32_F16S) {
        return HeadProfile::W8Full;
    }
    if (head_rows == detail::kLinearTopKFullRows && qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        return HeadProfile::Fp8Full;
    }
    if (head_rows == detail::kLinearTopKOptimizedRows && qtype == QType::Q4G64_F16S) {
        return HeadProfile::Q4Optimized;
    }
    throw std::invalid_argument("linear_topk: unsupported head profile");
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* label, std::uintptr_t alignment = 16) {
    if (tensor.dtype != dtype || tensor.ne[0] != rows || tensor.ne[1] != columns ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, alignment)) {
        throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
    }
}

void validate_io(const Tensor& hidden, const Tensor& candidate_ids,
                 const Tensor& candidate_scores) {
    if (hidden.dtype != DType::BF16 || hidden.ne[0] != detail::kLinearTopKHidden ||
        hidden.ne[1] <= 0 || hidden.ne[2] != 1 || hidden.ne[3] != 1 || !hidden.is_contiguous() ||
        !aligned_to(hidden.data, 16)) {
        throw std::invalid_argument("linear_topk: invalid hidden");
    }
    const std::int32_t columns = hidden.ne[1];
    const auto require_output  = [&](const Tensor& tensor, DType dtype, const char* label) {
        if (tensor.dtype != dtype || tensor.ne[0] != detail::kLinearTopK ||
            tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
            !tensor.is_contiguous() || !aligned_to(tensor.data, 16)) {
            throw std::invalid_argument(std::string("linear_topk: invalid ") + label);
        }
    };
    require_output(candidate_ids, DType::I32, "candidate_ids");
    require_output(candidate_scores, DType::FP32, "candidate_scores");
    if (overlaps(hidden, candidate_ids) || overlaps(hidden, candidate_scores) ||
        overlaps(candidate_ids, candidate_scores)) {
        throw std::invalid_argument("linear_topk: input and outputs must not overlap");
    }
}

void require_w8(const Weight& head) {
    const bool common =
        head.qtype == QType::W8G32_F16S && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 32 && head.group == 32 &&
        head.ndim == 2 && head.n == detail::kLinearTopKFullRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid W8 full head"); }
}

void require_q4(const Weight& head) {
    const bool common =
        head.qtype == QType::Q4G64_F16S && head.layout == QuantLayout::RowSplit &&
        head.scale_dtype == DType::FP16 && head.group_size == 64 && head.group == 64 &&
        head.ndim == 2 && head.n == detail::kLinearTopKOptimizedRows &&
        head.k == detail::kLinearTopKHidden && head.shape[0] == head.n && head.shape[1] == head.k &&
        head.padded_shape[0] == head.n && head.padded_shape[1] == head.k && head.qhigh == nullptr &&
        head.high_plane_bytes == 0 && aligned_to(head.qdata, 16) && aligned_to(head.scales, 16);
    if (!common) { throw std::invalid_argument("linear_topk: invalid Q4 optimized head"); }
}

Tensor column_slice(const Tensor& tensor, int first, int columns) {
    return Tensor(static_cast<std::uint8_t*>(tensor.data) +
                      static_cast<std::int64_t>(first) * tensor.nb[1],
                  tensor.dtype, {tensor.ne[0], columns});
}

void execute(const Tensor& hidden, const Weight& head, const Tensor* id_map, Tensor& ids,
             Tensor& scores, WorkspaceArena& workspace, cudaStream_t stream) {
    const auto profile = resolve_profile(head.qtype, head.n, head.k);
    const std::int32_t valid_rows = profile == HeadProfile::Q4Optimized
                                        ? detail::kLinearTopKOptimizedRows
                                        : detail::kLinearTopKFullValidRows;
#ifdef NINFER_VOLTA_BUILD
    // K=4..7 DFlash2 supplies 5..8 proposal columns, exactly the one-tile Volta QPN band. Emit
    // each 32-row CTA's top 16 order keys from the projection epilogue and merge those lists;
    // avoid both the dense BF16 logits and a second full-vocabulary read.
    if (profile == HeadProfile::Q4Optimized && hidden.ne[1] >= detail::kVoltaQpnMinT &&
        hidden.ne[1] <= detail::kVoltaQpnMaxT) {
        auto scope = workspace.scope();
        auto topk_workspace = detail::allocate_linear_topk_workspace(
            workspace, head.n, hidden.ne[1], detail::kVoltaQpnRowsPerTopKProducer);
        detail::launch_q4_volta_qpn_topk(
            hidden, head, *id_map,
            static_cast<std::uint64_t*>(topk_workspace.partial_keys.data),
            topk_workspace.producer_groups, stream);
        detail::linear_topk_merge_launch(topk_workspace, ids, scores, stream);
        return;
    }
#endif
    // sm_70 port: materialize the head logits with the general linear op, then a single
    // top-16 selection kernel. The BF16 logit scratch is a rounding the fused kernel avoids.
    for (int first = 0; first < hidden.ne[1];) {
        const int columns = std::min(detail::kLinearTopKMaxChunkColumns, hidden.ne[1] - first);
        auto x          = column_slice(hidden, first, columns);
        auto out_ids    = column_slice(ids, first, columns);
        auto out_scores = column_slice(scores, first, columns);
        auto scope      = workspace.scope();
        Tensor logits   = workspace.alloc(DType::BF16, {head.n, columns});
        linear(x, head, logits, stream);
        detail::dflash2_linear_topk16_launch(
            logits, id_map != nullptr ? static_cast<const std::int32_t*>(id_map->data) : nullptr,
            valid_rows, out_ids, out_scores, stream);
        first += columns;
    }
}
} // namespace

std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                 std::int32_t input_rows, std::int32_t min_columns,
                                                 std::int32_t max_columns) {
    (void)resolve_profile(qtype, head_rows, input_rows);
    if (min_columns < 1 || max_columns < min_columns) {
        throw std::invalid_argument("linear_topk workspace: invalid column interval");
    }
    const std::int32_t chunk = std::min(max_columns, detail::kLinearTopKMaxChunkColumns);
    std::size_t capacity = static_cast<std::size_t>(head_rows) * chunk * sizeof(std::uint16_t);
#ifdef NINFER_VOLTA_BUILD
    if (qtype == QType::Q4G64_F16S && head_rows == detail::kLinearTopKOptimizedRows &&
        min_columns <= detail::kVoltaQpnMaxT && max_columns >= detail::kVoltaQpnMinT) {
        const int columns = std::min(max_columns, detail::kVoltaQpnMaxT);
        WorkspaceLayoutBuilder layout;
        (void)detail::allocate_linear_topk_workspace(
            layout, head_rows, columns, detail::kVoltaQpnRowsPerTopKProducer);
        capacity = std::max(capacity, layout.peak_bytes());
    }
#endif
    return capacity;
}

void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    validate_io(hidden, candidate_ids, candidate_scores);
    const HeadProfile profile = resolve_profile(head.qtype, head.n, head.k);
    if (profile == HeadProfile::Q4Optimized || valid_rows != detail::kLinearTopKFullValidRows) {
        throw std::invalid_argument("linear_topk: invalid full-head profile or valid_rows");
    }
    if (profile == HeadProfile::W8Full) {
        require_w8(head);
    } else {
        (void)detail::validate_fp8_weight(head, "linear_topk FP8 full head");
    }

    execute(hidden, head, nullptr, candidate_ids, candidate_scores, workspace, stream);
}

void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    validate_io(hidden, candidate_ids, candidate_scores);
    if (resolve_profile(head.qtype, head.n, head.k) != HeadProfile::Q4Optimized) {
        throw std::invalid_argument("linear_topk: invalid optimized-head profile");
    }
    require_q4(head);
    require_matrix(row_to_global_ids, DType::I32, detail::kLinearTopKOptimizedRows, 1,
                   "row_to_global_ids", 4);
    if (overlaps(hidden, row_to_global_ids) || overlaps(candidate_ids, row_to_global_ids) ||
        overlaps(candidate_scores, row_to_global_ids)) {
        throw std::invalid_argument("linear_topk: id map overlaps input or output");
    }

    execute(hidden, head, &row_to_global_ids, candidate_ids, candidate_scores, workspace, stream);
}

} // namespace ninfer::ops
