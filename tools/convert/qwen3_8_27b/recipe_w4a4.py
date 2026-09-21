"""Closed single-source recipe for the Qwen3.8-27B NVFP4 W4A4 artifact.

Source: the Merkyor community NVFP4 export (ModelOpt 0.43, quant_algo NVFP4).
Field naming differs from the registered unsloth source:

    weight          -> packed U8 E2M1 codes [N, K/2]
    weight_scale    -> E4M3FN block scales [N, K/16]
    weight_scale_2  -> FP32[1] global weight scale, MULTIPLIER convention
    input_scale     -> FP32[1] global activation scale, MULTIPLIER convention

NInfer stores divisors, so both global words are written reciprocated:
``d_w = 1/weight_scale_2`` and ``d_x = 1/input_scale``.  ``materialize_nvfp4_weight``
validates this choice numerically per parent: the multiplier interpretation of
the stored word must yield sane weight magnitudes while the divisor
interpretation must not.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable

import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import recipe as official_recipe

from . import inventory_w4a4 as inventory


REPOSITORY = (
    "Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-MTP-NVFP4"
)
REVISION = "master"
RECIPE_ID = "qwen3_8_27b_nvfp4_w4a4-v1"

PACKED_FIELD = "weight"
SCALE_FIELD = "weight_scale"
WEIGHT_GLOBAL_FIELD = "weight_scale_2"
INPUT_GLOBAL_FIELD = "input_scale"


@dataclass(frozen=True, slots=True)
class RowRange:
    begin: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True, slots=True)
class MatrixSource:
    name: str
    shape: tuple[int, int]

    def field(self, suffix: str) -> str:
        return f"{self.name}.{suffix}"


@dataclass(frozen=True, slots=True)
class MatrixPart:
    source: MatrixSource
    rows: tuple[RowRange, ...]

    @property
    def output_rows(self) -> int:
        return sum(item.rows for item in self.rows)


@dataclass(frozen=True, slots=True)
class Nvfp4WeightRecipe:
    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]
    divisor_sources: tuple[MatrixSource, ...]


@dataclass(frozen=True, slots=True)
class InputDivisorRecipe:
    object_name: str
    sources: tuple[MatrixSource, ...]
    weight_names: tuple[str, ...]


def _source(name: str, n: int, k: int) -> MatrixSource:
    return MatrixSource(name, (n, k))


def _all(source: MatrixSource) -> MatrixPart:
    return MatrixPart(source, (RowRange(0, source.shape[0]),))


def _q_part(source: MatrixSource, gate: bool) -> MatrixPart:
    begin = 256 if gate else 0
    return MatrixPart(
        source,
        tuple(
            RowRange(head * 512 + begin, head * 512 + begin + 256)
            for head in range(24)
        ),
    )


def _build_nvfp4_matrix_recipes() -> tuple[
    tuple[Nvfp4WeightRecipe, ...],
    tuple[InputDivisorRecipe, ...],
]:
    nvfp4_weights: list[Nvfp4WeightRecipe] = []
    input_divisors: list[InputDivisorRecipe] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        if layer in inventory.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            fused_sources = (query, key, value)
            nvfp4_weights.extend(
                (
                    Nvfp4WeightRecipe(
                        object_prefix + "attention/query_key_gate_value",
                        (14336, 5120),
                        (
                            _q_part(query, False),
                            _all(key),
                            _q_part(query, True),
                            _all(value),
                        ),
                        fused_sources,
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "attention/output",
                        output.shape,
                        (_all(output),),
                        (output,),
                    ),
                )
            )
            input_divisors.extend(
                (
                    InputDivisorRecipe(
                        object_prefix
                        + "attention/input_projection/input_scale_divisor",
                        fused_sources,
                        (object_prefix + "attention/query_key_gate_value",),
                    ),
                    InputDivisorRecipe(
                        object_prefix
                        + "attention/output_projection/input_scale_divisor",
                        (output,),
                        (object_prefix + "attention/output",),
                    ),
                )
            )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            output = _source(
                source_prefix + "linear_attn.out_proj", 5120, 6144
            )
            fused_sources = (query_key_value, z)
            nvfp4_weights.extend(
                (
                    Nvfp4WeightRecipe(
                        object_prefix + "gdn/query_key_value_z",
                        (16384, 5120),
                        (_all(query_key_value), _all(z)),
                        fused_sources,
                    ),
                    Nvfp4WeightRecipe(
                        object_prefix + "gdn/output",
                        output.shape,
                        (_all(output),),
                        (output,),
                    ),
                )
            )
            input_divisors.extend(
                (
                    InputDivisorRecipe(
                        object_prefix
                        + "gdn/input_projection/input_scale_divisor",
                        fused_sources,
                        (object_prefix + "gdn/query_key_value_z",),
                    ),
                    InputDivisorRecipe(
                        object_prefix
                        + "gdn/output_projection/input_scale_divisor",
                        (output,),
                        (object_prefix + "gdn/output",),
                    ),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        gate_up_sources = (gate, up)
        nvfp4_weights.extend(
            (
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/gate_up",
                    (34816, 5120),
                    (_all(gate), _all(up)),
                    gate_up_sources,
                ),
                Nvfp4WeightRecipe(
                    object_prefix + "mlp/down",
                    down.shape,
                    (_all(down),),
                    (down,),
                ),
            )
        )
        input_divisors.extend(
            (
                InputDivisorRecipe(
                    object_prefix + "mlp/gate_up_projection/input_scale_divisor",
                    gate_up_sources,
                    (object_prefix + "mlp/gate_up",),
                ),
                InputDivisorRecipe(
                    object_prefix + "mlp/down_projection/input_scale_divisor",
                    (down,),
                    (object_prefix + "mlp/down",),
                ),
            )
        )

    return tuple(nvfp4_weights), tuple(input_divisors)


(
    NVFP4_WEIGHT_RECIPES,
    INPUT_DIVISOR_RECIPES,
) = _build_nvfp4_matrix_recipes()
NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in NVFP4_WEIGHT_RECIPES
}
INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in INPUT_DIVISOR_RECIPES
}

NVFP4_SOURCES = tuple(
    dict.fromkeys(
        part.source for recipe in NVFP4_WEIGHT_RECIPES for part in recipe.parts
    )
)


def _build_quantized_direct_recipes() -> tuple[family_recipe.TensorRecipe, ...]:
    recipes: list[family_recipe.TensorRecipe] = []
    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            family_recipe.TensorRecipe(
                object_prefix + "input_norm",
                family_recipe.source(
                    source_prefix + "input_layernorm.weight", (5120,)
                ),
            )
        )
        if layer in inventory.FULL_ATTENTION_LAYERS:
            recipes.extend(
                (
                    family_recipe.TensorRecipe(
                        object_prefix + "attention/query_norm",
                        family_recipe.source(
                            source_prefix + "self_attn.q_norm.weight", (256,)
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "attention/key_norm",
                        family_recipe.source(
                            source_prefix + "self_attn.k_norm.weight", (256,)
                        ),
                    ),
                )
            )
        else:
            convolution = family_recipe.source(
                source_prefix + "linear_attn.conv1d.weight", (10240, 1, 4)
            )
            recipes.extend(
                (
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/a_log",
                        family_recipe.Cast(
                            family_recipe.source(
                                source_prefix + "linear_attn.A_log", (48,)
                            ),
                            inventory.FP32,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        family_recipe.Cast(
                            family_recipe.source(
                                source_prefix + "linear_attn.dt_bias", (48,)
                            ),
                            inventory.FP32,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/convolution",
                        family_recipe.Transpose(
                            family_recipe.Reshape(
                                family_recipe.Slice(convolution, 1, 0, 1),
                                (10240, 4),
                            ),
                            (1, 0),
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/a_b_projection",
                        family_recipe.Concat(
                            (
                                family_recipe.source(
                                    source_prefix
                                    + "linear_attn.in_proj_a.weight",
                                    (48, 5120),
                                ),
                                family_recipe.source(
                                    source_prefix
                                    + "linear_attn.in_proj_b.weight",
                                    (48, 5120),
                                ),
                            ),
                            0,
                        ),
                    ),
                    family_recipe.TensorRecipe(
                        object_prefix + "gdn/norm",
                        family_recipe.source(
                            source_prefix + "linear_attn.norm.weight", (128,)
                        ),
                    ),
                )
            )
        recipes.append(
            family_recipe.TensorRecipe(
                object_prefix + "post_attention_norm",
                family_recipe.source(
                    source_prefix + "post_attention_layernorm.weight", (5120,)
                ),
            )
        )
    recipes.append(
        family_recipe.TensorRecipe(
            "text/final_norm",
            family_recipe.source("model.language_model.norm.weight", (5120,)),
        )
    )
    return tuple(recipes)


QUANTIZED_DIRECT_RECIPES = _build_quantized_direct_recipes()
QUANTIZED_DIRECT_BY_NAME = {
    item.object_name: item for item in QUANTIZED_DIRECT_RECIPES
}
QUANTIZED_DIRECT_SPECS = tuple(
    spec
    for spec in inventory.TEXT_CORE_TENSOR_SPECS
    if spec.name in QUANTIZED_DIRECT_BY_NAME
)

# Official-route objects are materialized from the BF16 tensors carried inside
# the same community checkpoint (embedding, lm_head, MTP, Vision, draft head).
OFFICIAL_TENSOR_SPECS = tuple(
    spec
    for spec in inventory.BASE_TENSOR_SPECS
    if spec.name == "text/token_embedding"
    or spec.name == "text/output_head"
    or spec.name.startswith("text/draft_head")
    or spec.name.startswith("mtp/")
    or spec.name.startswith("vision/")
)
OFFICIAL_RECIPES = tuple(
    official_recipe.RECIPES_BY_NAME[spec.name]
    for spec in OFFICIAL_TENSOR_SPECS
)
OFFICIAL_RECIPES_BY_NAME = {
    item.object_name: item for item in OFFICIAL_RECIPES
}


def validate_recipe() -> None:
    family_recipe.validate_recipe_coverage(
        QUANTIZED_DIRECT_RECIPES, QUANTIZED_DIRECT_SPECS
    )
    family_recipe.validate_recipe_coverage(
        OFFICIAL_RECIPES, OFFICIAL_TENSOR_SPECS
    )
    ownership = (
        set(NVFP4_WEIGHTS_BY_NAME),
        set(INPUT_DIVISORS_BY_NAME),
        set(QUANTIZED_DIRECT_BY_NAME),
        set(OFFICIAL_RECIPES_BY_NAME),
    )
    all_names: set[str] = set()
    for names in ownership:
        if all_names.intersection(names):
            raise ValueError("more than one source route owns an artifact tensor")
        all_names.update(names)
    if all_names != {spec.name for spec in inventory.BASE_TENSOR_SPECS}:
        raise ValueError("source routes do not cover the base tensor inventory")
    if tuple(NVFP4_WEIGHTS_BY_NAME) != tuple(
        spec.name for spec in inventory.NVFP4_TENSOR_SPECS
    ):
        raise ValueError("NVFP4 recipe order does not match inventory")
    if tuple(INPUT_DIVISORS_BY_NAME) != tuple(
        spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS
    ):
        raise ValueError("input-divisor recipe order does not match inventory")
    for recipe in NVFP4_WEIGHT_RECIPES:
        _validate_matrix_recipe(recipe.object_name, recipe.shape, recipe.parts)
    bound_weights = tuple(
        name for site in INPUT_DIVISOR_RECIPES for name in site.weight_names
    )
    if (
        len(bound_weights) != len(set(bound_weights))
        or set(bound_weights) != set(NVFP4_WEIGHTS_BY_NAME)
    ):
        raise ValueError("input-divisor sites do not cover NVFP4 parents once")


def _validate_matrix_recipe(
    object_name: str,
    shape: tuple[int, int],
    parts: tuple[MatrixPart, ...],
) -> None:
    rows = sum(part.output_rows for part in parts)
    if not parts or (rows, parts[0].source.shape[1]) != shape:
        raise ValueError(f"{object_name}: invalid fused row geometry")
    if any(part.source.shape[1] != shape[1] for part in parts):
        raise ValueError(f"{object_name}: incompatible source K")


def _merge_requirement(
    result: dict[str, tuple[tuple[int, ...], str]],
    name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    signature = (shape, dtype)
    previous = result.setdefault(name, signature)
    if previous != signature:
        raise ValueError(f"inconsistent source declaration for {name}")


def _source_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in NVFP4_SOURCES:
        n, k = source.shape
        _merge_requirement(result, source.field(PACKED_FIELD), (n, k // 2), "U8")
        _merge_requirement(
            result, source.field(SCALE_FIELD), (n, k // 16), "F8_E4M3"
        )
        _merge_requirement(
            result, source.field(WEIGHT_GLOBAL_FIELD), (1,), "F32"
        )
        _merge_requirement(
            result, source.field(INPUT_GLOBAL_FIELD), (1,), "F32"
        )
    for source in family_recipe.source_requirements(
        QUANTIZED_DIRECT_RECIPES
    ).values():
        _merge_requirement(result, source.name, source.shape, source.dtype)
    return result


SOURCE_REQUIREMENTS = _source_requirements()


def preflight_source_metadata(
    reader: ShardReader,
) -> family_recipe.SourcePreflight:
    missing = set(SOURCE_REQUIREMENTS).difference(reader.names)
    if missing:
        raise ValueError(f"quantized source is missing {sorted(missing)[0]}")
    metadata = reader.metadata(sorted(SOURCE_REQUIREMENTS))
    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, (shape, dtype) in SOURCE_REQUIREMENTS.items():
        actual = metadata[name]
        # Scalar globals are stored as shape [] in the ModelOpt 0.43 export
        # and as [1] in the registered unsloth export; both mean FP32[1].
        allowed = (shape, ()) if shape == (1,) else (shape,)
        if actual.dtype != dtype or actual.shape not in allowed:
            raise ValueError(
                f"{name}: source signature {(actual.shape, actual.dtype)} "
                f"!= {(shape, dtype)}"
            )
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        shards.add(actual.shard)
    return family_recipe.SourcePreflight(
        recipe_count=(
            len(NVFP4_WEIGHT_RECIPES)
            + len(INPUT_DIVISOR_RECIPES)
            + len(QUANTIZED_DIRECT_RECIPES)
        ),
        source_tensor_count=len(SOURCE_REQUIREMENTS),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def _f32_word(tensor: torch.Tensor, name: str) -> int:
    if tensor.dtype != torch.float32 or tensor.numel() != 1:
        raise ValueError(f"{name}: global scale must be FP32[1]")
    word = int(tensor.detach().contiguous().cpu().view(torch.int32).item())
    word &= 0xFFFFFFFF
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{name}: global scale must be finite and positive")
    return word


def _same_word(
    reader: ShardReader,
    sources: Iterable[MatrixSource],
    suffix: str,
) -> int:
    items = tuple(sources)
    words = tuple(
        _f32_word(reader.get(source.field(suffix)), source.field(suffix))
        for source in items
    )
    if len(set(words)) != 1:
        raise ValueError(f"{items[0].name}: fused {suffix} words differ")
    return words[0]


def _reciprocal_bits(value: float, label: str) -> int:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label}: global scale must be finite and positive")
    reciprocal = 1.0 / value
    if not math.isfinite(reciprocal) or reciprocal <= 0.0:
        raise ValueError(f"{label}: reciprocal scale overflowed")
    return struct.unpack("<I", struct.pack("<f", reciprocal))[0]


def _decode_row(packed: memoryview, scales: memoryview, columns: int) -> list[float]:
    e2m1_lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    def e2m1(code: int) -> float:
        return e2m1_lut[code & 7] * (1 - 2 * ((code >> 3) & 1))

    def e4m3(byte: int) -> float:
        sign = 1 - 2 * ((byte >> 7) & 1)
        exponent = (byte >> 3) & 15
        mantissa = byte & 7
        if exponent == 0:
            return sign * mantissa * 2.0**-9
        return sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7)

    values: list[float] = []
    for group in range(columns // 16):
        scale = e4m3(scales[group])
        base = group * 8
        for byte in packed[base:base + 8]:
            values.append(e2m1(byte & 15) * scale)
            values.append(e2m1(byte >> 4) * scale)
    return values


def _validate_global_word_choice(
    source: MatrixSource,
    packed: torch.Tensor,
    scales: torch.Tensor,
    global_word: float,
) -> None:
    """The stored global word must be a multiplier, not a divisor.

    Decodes row 0 both ways and requires the multiplier interpretation to give
    sane weight magnitudes while the divisor interpretation does not.
    """
    row = _decode_row(
        memoryview(packed[0].contiguous().numpy().tobytes()),
        memoryview(scales[0].contiguous().view(torch.uint8).numpy().tobytes()),
        source.shape[1],
    )
    code_scale_std = math.sqrt(
        sum(value * value for value in row) / len(row)
    )
    as_multiplier = code_scale_std * global_word
    as_divisor = code_scale_std / global_word
    if not 1e-4 <= as_multiplier <= 10.0:
        raise ValueError(
            f"{source.name}: multiplier interpretation std {as_multiplier:.4g} "
            "is outside the sane weight range; source convention changed"
        )
    if 1e-4 <= as_divisor <= 10.0:
        raise ValueError(
            f"{source.name}: both interpretations give sane magnitudes "
            f"(multiplier {as_multiplier:.4g}, divisor {as_divisor:.4g}); "
            "convention is ambiguous"
        )


def _select_rows(tensor: torch.Tensor, part: MatrixPart) -> torch.Tensor:
    pieces = [
        tensor.narrow(0, row_range.begin, row_range.rows)
        for row_range in part.rows
    ]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)


class _SourceWords:
    """Cached packed-code and block-scale planes for one source matrix."""

    def __init__(self, reader: ShardReader, source: MatrixSource) -> None:
        n, k = source.shape
        packed = reader.get(source.field(PACKED_FIELD))
        scales = reader.get(source.field(SCALE_FIELD))
        if (
            packed.dtype != torch.uint8
            or tuple(packed.shape) != (n, k // 2)
            or scales.dtype != torch.float8_e4m3fn
            or tuple(scales.shape) != (n, k // 16)
        ):
            raise ValueError(
                f"{source.name}: NVFP4 source signature mismatch"
            )
        self.packed = packed
        self.scales = scales.view(torch.uint8)


def materialize_nvfp4_weight(
    recipe: Nvfp4WeightRecipe,
    reader: ShardReader,
    words_cache: dict[MatrixSource, _SourceWords],
) -> tuple[torch.Tensor, torch.Tensor, bytes]:
    packed_parts: list[torch.Tensor] = []
    scale_parts: list[torch.Tensor] = []
    for part in recipe.parts:
        words = words_cache.get(part.source)
        if words is None:
            words = _SourceWords(reader, part.source)
            words_cache[part.source] = words
        packed_parts.append(_select_rows(words.packed, part))
        scale_parts.append(_select_rows(words.scales, part))
    packed = (
        packed_parts[0].contiguous()
        if len(packed_parts) == 1
        else torch.cat(packed_parts, dim=0)
    )
    scales = (
        scale_parts[0].contiguous()
        if len(scale_parts) == 1
        else torch.cat(scale_parts, dim=0)
    )
    global_bits = _same_word(
        reader, recipe.divisor_sources, WEIGHT_GLOBAL_FIELD
    )
    global_value = struct.unpack("<f", struct.pack("<I", global_bits))[0]
    _validate_global_word_choice(
        recipe.divisor_sources[0], packed, scales, global_value
    )
    divisor_bits = _reciprocal_bits(
        global_value, recipe.divisor_sources[0].name
    )
    if tuple(packed.shape) != (recipe.shape[0], recipe.shape[1] // 2) or tuple(
        scales.shape
    ) != (recipe.shape[0], recipe.shape[1] // 16):
        raise ValueError(
            f"{recipe.object_name}: materialized NVFP4 shape mismatch"
        )
    return packed, scales, struct.pack("<I", divisor_bits)


def materialize_input_divisor(
    recipe: InputDivisorRecipe,
    reader: ShardReader,
) -> torch.Tensor:
    bits = _same_word(reader, recipe.sources, INPUT_GLOBAL_FIELD)
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    divisor_bits = _reciprocal_bits(value, recipe.sources[0].name)
    return torch.frombuffer(
        bytearray(struct.pack("<I", divisor_bits)), dtype=torch.float32
    ).reshape(())


def materialize_quantized_direct(
    object_name: str,
    reader: ShardReader,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        QUANTIZED_DIRECT_BY_NAME[object_name], reader
    )


def materialize_official(
    object_name: str,
    reader: ShardReader,
    derived_tensors: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    return family_recipe.materialize_recipe(
        OFFICIAL_RECIPES_BY_NAME[object_name], reader, derived_tensors
    )


validate_recipe()


__all__ = [
    "INPUT_DIVISOR_RECIPES",
    "INPUT_DIVISORS_BY_NAME",
    "NVFP4_SOURCES",
    "NVFP4_WEIGHT_RECIPES",
    "NVFP4_WEIGHTS_BY_NAME",
    "OFFICIAL_RECIPES",
    "OFFICIAL_RECIPES_BY_NAME",
    "OFFICIAL_TENSOR_SPECS",
    "QUANTIZED_DIRECT_BY_NAME",
    "QUANTIZED_DIRECT_RECIPES",
    "QUANTIZED_DIRECT_SPECS",
    "RECIPE_ID",
    "REPOSITORY",
    "REVISION",
    "SOURCE_REQUIREMENTS",
    "materialize_input_divisor",
    "materialize_nvfp4_weight",
    "materialize_official",
    "materialize_quantized_direct",
    "preflight_source_metadata",
    "validate_recipe",
]
