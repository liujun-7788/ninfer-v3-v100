#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/layout.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct RouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8LinearSwiGluScheduleId schedule;
};

#ifdef NINFER_VOLTA_BUILD
// The DFlash MLP is W8-only and its tuned routes all use Ampere+ MMA past
// decode. The paired SIMT kernel streams gate/up rows together and tiles any T.
constexpr std::array<RouteSpec, 3> kRoutes{{
    {1, 1, W8LinearSwiGluScheduleId::DecodePairR16},
    {2, 4, W8LinearSwiGluScheduleId::SimtPairC4},
    {5, kAnyCols, W8LinearSwiGluScheduleId::SimtPairC8},
}};
#else
constexpr std::array<RouteSpec, 18> kRoutes{{
    {1, 1, W8LinearSwiGluScheduleId::DecodePairR16},
    {2, 48, W8LinearSwiGluScheduleId::SplitKMmaExactT},
    {49, 64, W8LinearSwiGluScheduleId::MmaR32C64},
    {65, 80, W8LinearSwiGluScheduleId::MmaR32C80},
    {81, 96, W8LinearSwiGluScheduleId::MmaR32C96},
    {97, 128, W8LinearSwiGluScheduleId::MmaR64C64},
    {129, 192, W8LinearSwiGluScheduleId::MmaR32C64},
    {193, 240, W8LinearSwiGluScheduleId::MmaR128C80},
    {241, 255, W8LinearSwiGluScheduleId::MmaR32C128},
    {256, 256, W8LinearSwiGluScheduleId::MmaR64C128},
    {257, 264, W8LinearSwiGluScheduleId::MmaR64C64},
    {265, 288, W8LinearSwiGluScheduleId::MmaR64C96},
    {289, 320, W8LinearSwiGluScheduleId::MmaR64C64},
    {321, 384, W8LinearSwiGluScheduleId::MmaR64C128},
    {385, 448, W8LinearSwiGluScheduleId::MmaR128C64},
    {449, 512, W8LinearSwiGluScheduleId::MmaR64C128},
    {513, 560, W8LinearSwiGluScheduleId::MmaR128C80},
    {561, kAnyCols, W8LinearSwiGluScheduleId::MmaR64C128},
}};
#endif

constexpr bool catalog_is_closed() {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.first != expected || route.first > route.last) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "W8 LinearSwiGLU routes must be exact and closed");

constexpr bool is_dflash1_shape(const W8LinearSwiGluProblem& p) noexcept {
    return p.gate_up_rows == 12288 && p.output_rows == 6144 && p.k == 2048 && p.padded_k == 2048;
}

constexpr bool is_dflash2_shape(const W8LinearSwiGluProblem& p) noexcept {
    return p.gate_up_rows == 34816 && p.output_rows == 17408 && p.k == 5120 &&
           p.padded_k == 5120;
}

bool supported_shape(const W8LinearSwiGluProblem& problem) noexcept {
    return is_dflash1_shape(problem) || is_dflash2_shape(problem);
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows,
                                       std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

} // namespace

const char* w8_linear_swiglu_schedule_name(W8LinearSwiGluScheduleId schedule) noexcept {
    if (schedule == W8LinearSwiGluScheduleId::Materialized) {
        return "linear_swiglu.w8.materialized";
    }
    switch (schedule) {
    case W8LinearSwiGluScheduleId::DecodePairR16:
        return "linear_swiglu.w8.decode.pair.r16";
    case W8LinearSwiGluScheduleId::SimtPairC4:
        return "linear_swiglu.w8.simt.pair.c4";
    case W8LinearSwiGluScheduleId::SimtPairC8:
        return "linear_swiglu.w8.simt.pair.c8";
    case W8LinearSwiGluScheduleId::VoltaQpnSplit:
        return "linear_swiglu.w8.sm70.qpn.split";
    case W8LinearSwiGluScheduleId::SplitKMmaExactT:
        return "linear_swiglu.w8.splitk.mma.pair.exact_t";
    case W8LinearSwiGluScheduleId::MmaR32C64:
        return "linear_swiglu.w8.mma.pair.r16.c64";
    case W8LinearSwiGluScheduleId::MmaR32C80:
        return "linear_swiglu.w8.mma.pair.r16.c80";
    case W8LinearSwiGluScheduleId::MmaR32C96:
        return "linear_swiglu.w8.mma.pair.r16.c96";
    case W8LinearSwiGluScheduleId::MmaR32C128:
        return "linear_swiglu.w8.mma.pair.r16.c128";
    case W8LinearSwiGluScheduleId::MmaR64C64:
        return "linear_swiglu.w8.mma.pair.r32.c64";
    case W8LinearSwiGluScheduleId::MmaR64C96:
        return "linear_swiglu.w8.mma.pair.r32.c96";
    case W8LinearSwiGluScheduleId::MmaR64C128:
        return "linear_swiglu.w8.mma.pair.r32.c128";
    case W8LinearSwiGluScheduleId::MmaR128C64:
        return "linear_swiglu.w8.mma.pair.r64.c64";
    case W8LinearSwiGluScheduleId::MmaR128C80:
        return "linear_swiglu.w8.mma.pair.r64.c80";
    }
    return "linear_swiglu.w8.unknown";
}

bool w8_linear_swiglu_schedule_uses_mma(W8LinearSwiGluScheduleId schedule) noexcept {
    return schedule != W8LinearSwiGluScheduleId::DecodePairR16 &&
           schedule != W8LinearSwiGluScheduleId::SimtPairC4 &&
           schedule != W8LinearSwiGluScheduleId::SimtPairC8 &&
           schedule != W8LinearSwiGluScheduleId::Materialized;
}

std::size_t w8_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    const W8LinearSwiGluProblem lo{gate_up_rows, output_rows, k, padded_k, min_tokens};
    const W8LinearSwiGluProblem hi{gate_up_rows, output_rows, k, padded_k, max_tokens};
    (void)w8_linear_swiglu_resolve_plan(lo);
    (void)w8_linear_swiglu_resolve_plan(hi);
    if (!is_dflash2_shape(hi)) { return 0; }
#ifdef NINFER_VOLTA_BUILD
    if (min_tokens >= 5 && max_tokens <= 8) { return 256; }
#endif
    return materialized_workspace_bytes(gate_up_rows, max_tokens);
}

bool w8_linear_swiglu_admits(const W8LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols > 0;
}

W8LinearSwiGluPlan w8_linear_swiglu_resolve_plan(const W8LinearSwiGluProblem& problem) {
    if (!w8_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "W8 LinearSwiGLU: exact problem or column count is not admitted");
    }
    if (is_dflash2_shape(problem)) {
#ifdef NINFER_VOLTA_BUILD
        if (problem.cols >= 5 && problem.cols <= 8) {
            return {W8LinearSwiGluScheduleId::VoltaQpnSplit};
        }
#endif
        return {W8LinearSwiGluScheduleId::Materialized};
    }
    for (const RouteSpec& route : kRoutes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("W8 LinearSwiGLU: admitted problem has no route");
}

void w8_linear_swiglu_execute_plan(const W8LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, WorkspaceArena& ws, cudaStream_t stream) {
    const W8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const W8LinearSwiGluPlan resolved = w8_linear_swiglu_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("W8 LinearSwiGLU: plan does not match exact problem");
    }
    if (plan.schedule == W8LinearSwiGluScheduleId::Materialized) {
        auto scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    if (plan.schedule == W8LinearSwiGluScheduleId::VoltaQpnSplit) {
        auto scope = ws.scope();
        (void)ws.alloc_bytes(256);
        w8_linear_swiglu_volta_qpn_split_launch(x, w, out, stream);
        return;
    }
    switch (plan.schedule) {
    case W8LinearSwiGluScheduleId::DecodePairR16:
        w8_linear_swiglu_decode_pair_r16_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::SimtPairC4:
        w8_linear_swiglu_simt_pair_c4_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::SimtPairC8:
        w8_linear_swiglu_simt_pair_c8_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::VoltaQpnSplit:
        break; // handled above
    case W8LinearSwiGluScheduleId::SplitKMmaExactT:
        w8_linear_swiglu_splitk_exact_t_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C64:
        w8_linear_swiglu_mma_r32_c64_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C80:
        w8_linear_swiglu_mma_r32_c80_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C96:
        w8_linear_swiglu_mma_r32_c96_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR32C128:
        w8_linear_swiglu_mma_r32_c128_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C64:
        w8_linear_swiglu_mma_r64_c64_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C96:
        w8_linear_swiglu_mma_r64_c96_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR64C128:
        w8_linear_swiglu_mma_r64_c128_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR128C64:
        w8_linear_swiglu_mma_r128_c64_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::MmaR128C80:
        w8_linear_swiglu_mma_r128_c80_launch(x, w, out, stream);
        return;
    case W8LinearSwiGluScheduleId::Materialized:
        break; // handled above
    }
    throw std::logic_error("W8 LinearSwiGLU: unknown schedule");
}

void w8_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                               cudaStream_t stream) {
    const W8LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    w8_linear_swiglu_execute_plan(w8_linear_swiglu_resolve_plan(problem), x, w, out, ws, stream);
}

} // namespace ninfer::ops::detail
