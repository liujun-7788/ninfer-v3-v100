"""Build the experimental Qwen3.8-27B NVFP4 mixed artifact from the Merkyor
``W4A4+W8A8`` checkpoint and the registered DFlash2 companion.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_w4a8 \
      --model /path/to/merkyor-w4a8/W4A4+W8A8 \
      --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
      --out out/qwen3_8_27b_nvfp4_mixed.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.layouts import (
    encode_direct,
    encode_fp8_row_scaled,
    encode_nvfp4,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head
from tools.convert.qwen3_8_27b import fp8_embedding

from . import dflash2_recipe
from . import inventory_w4a8 as inventory
from . import recipe_w4a8 as recipe
from .dflash2_inventory import DFLASH2_TENSOR_SPECS


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    dflash2_model_dir: Path
    base_config_summary: dict[str, object]
    dflash2_config_summary: dict[str, object]
    source: family_recipe.SourcePreflight
    dflash2_source: family_recipe.SourcePreflight
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(
            f"{model_dir}: safetensors shard set does not match the index"
        )
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def preflight_conversion(
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
) -> ConversionPreflight:
    model = Path(model_dir)
    dflash2_model = Path(dflash2_model_dir)
    _validate_index(model)

    config = family_conversion.load_json(model / "config.json")
    base_summary = family_config.validate_config(config)
    dflash2_summary = dflash2_recipe.validate_config(
        family_conversion.load_json(dflash2_model / "config.json")
    )
    dflash2_recipe.validate_base_compatibility(base_summary, dflash2_summary)
    inventory.validate_inventory()
    recipe.validate_recipe()
    dflash2_recipe.validate_recipe_coverage()

    with ShardReader(model) as reader:
        source = recipe.preflight_source_metadata(reader)
    dflash2_source = dflash2_recipe.preflight_sources(dflash2_model)

    # Community checkpoint: load the same six frontend resources without the
    # official-source SHA256 pinning applied by the stock loader.
    resources = family_conversion.load_resources(model, inventory.RESOURCE_SPECS)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = family_conversion.build_object_plan(
        inventory.OBJECT_SPECS, resource_map
    )
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, model)
    return ConversionPreflight(
        model_dir=model,
        dflash2_model_dir=dflash2_model,
        base_config_summary=base_summary,
        dflash2_config_summary=dflash2_summary,
        source=source,
        dflash2_source=dflash2_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _encode_nvfp4_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    words_cache: dict,
) -> bytes:
    selected = recipe.NVFP4_WEIGHTS_BY_NAME[spec.name]
    packed, scales, divisor = recipe.materialize_nvfp4_weight(
        selected, reader, words_cache
    )
    return encode_nvfp4(packed, scales, divisor, spec.shape)


def _encode_fp8_weight(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> bytes:
    selected = recipe.FP8_WEIGHTS_BY_NAME[spec.name]
    codes, scales = recipe.materialize_fp8_weight(selected, reader)
    return encode_fp8_row_scaled(codes, scales, spec.shape)


def _encode_fp8_from_bf16(
    reader: ShardReader,
    source_name: str,
    spec: inventory.TensorSpec,
) -> Iterable[bytes]:
    return fp8_embedding.iter_reader_payload(reader, source_name, spec.shape)


def _materialize_direct(
    spec: inventory.TensorSpec,
    reader: ShardReader,
) -> torch.Tensor:
    tensor = recipe.materialize_quantized_direct(spec.name, reader)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _materialize_official(
    spec: inventory.TensorSpec,
    reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    tensor = recipe.materialize_official(spec.name, reader, dict(derived))
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=recipe.RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.model_dir,
        out_path=output,
        arguments=arguments,
        config_summary={
            "base": preflight.base_config_summary,
            "dflash2": preflight.dflash2_config_summary,
        },
        source_preflight=preflight.source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "community_quantized": {
            "repository": recipe.REPOSITORY,
            "revision": recipe.REVISION,
            "model_path": str(preflight.model_dir.resolve()),
            "convention_notes": (
                "ModelOpt 0.43 export: FP8 per-tensor scales are multipliers "
                "written as one BF16 row scale per row; NVFP4 global scales "
                "are multipliers stored reciprocated to match the ninfer "
                "divisor convention"
            ),
        },
        "dflash2": {
            "repository": dflash2_recipe.REPOSITORY,
            "revision": dflash2_recipe.REVISION,
            "model_path": str(preflight.dflash2_model_dir.resolve()),
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["source_preflight"] = {
        "community_quantized": {
            "recipes": preflight.source.recipe_count,
            "tensors": preflight.source.source_tensor_count,
            "shards": preflight.source.source_shard_count,
            "dtypes": dict(preflight.source.source_dtype_counts),
        },
        "dflash2": {
            "recipes": preflight.dflash2_source.recipe_count,
            "tensors": preflight.dflash2_source.source_tensor_count,
            "shards": preflight.dflash2_source.source_shard_count,
            "dtypes": dict(preflight.dflash2_source.source_dtype_counts),
        },
    }
    return report


def convert(
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the closed two-source conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model_dir, dflash2_model_dir)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.FP8_SOURCES)} FP8 and "
        f"{len(recipe.NVFP4_SOURCES)} NVFP4 source matrices, "
        f"{preflight.dflash2_source.source_tensor_count} DFlash2 source tensors, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    total = len(inventory.OBJECT_SPECS)
    index = 0
    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        preflight.object_plan.specs,
    ) as writer:
        if writer.objects != preflight.object_plan.objects:
            raise RuntimeError(
                "writer object plan differs from completed preflight"
            )

        for spec in inventory.RESOURCE_SPECS:
            index += 1
            writer.write(spec.name, resources[spec.name])
            print(f"[{index}/{total}] {spec.name}", flush=True)

        with ShardReader(preflight.model_dir) as reader:
            words_cache: dict = {}
            for spec in inventory.BASE_TENSOR_SPECS:
                index += 1
                payload: bytes | Iterable[bytes]
                if spec.name == "text/token_embedding":
                    payload = _encode_fp8_from_bf16(
                        reader, recipe.EMBEDDING_SOURCE, spec
                    )
                elif spec.name == "text/output_head":
                    payload = _encode_fp8_from_bf16(
                        reader, recipe.OUTPUT_HEAD_SOURCE, spec
                    )
                elif spec.name in recipe.FP8_WEIGHTS_BY_NAME:
                    payload = _encode_fp8_weight(spec, reader)
                elif spec.name in recipe.NVFP4_WEIGHTS_BY_NAME:
                    payload = _encode_nvfp4_weight(spec, reader, words_cache)
                elif spec.name in recipe.INPUT_DIVISORS_BY_NAME:
                    scalar = recipe.materialize_input_divisor(
                        recipe.INPUT_DIVISORS_BY_NAME[spec.name], reader
                    )
                    payload = encode_direct(scalar, inventory.FP32)
                elif spec.name in recipe.QUANTIZED_DIRECT_BY_NAME:
                    tensor = _materialize_direct(spec, reader)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                else:
                    tensor = _materialize_official(spec, reader, derived)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                if index % 64 == 0:
                    print(f"[{index}/{total}] objects written", flush=True)

        with ShardReader.from_file(
            preflight.dflash2_model_dir / "model.safetensors"
        ) as dflash2_reader:
            for spec in DFLASH2_TENSOR_SPECS:
                index += 1
                tensor = dflash2_recipe.materialize_tensor(
                    spec.name, dflash2_reader
                )
                payload = family_conversion.encode_tensor_payload(
                    tensor, spec, resolved_device
                )
                del tensor
                writer.write(spec.name, payload)
                del payload

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(model_dir),
        "dflash2_model": str(dflash2_model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(
        arguments.model,
        arguments.dflash2_model,
        arguments.out,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
