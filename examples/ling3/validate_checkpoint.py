#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""Validate the pinned Ling-3.0-tiny source contract and W4A32 recipe."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


PINNED_MODEL_ID = "inclusionAI/Ling-3.0-tiny"
PINNED_REVISION = "a2ee06c0f2de5b171701aee7f73f70a1da75483b"
FULL_ATTENTION_LAYERS = {3, 7, 11, 15, 19, 23}
OFFICIAL_CONTRACT = {
    "architectures": ["BailingMoeV3ForCausalLM"],
    "model_type": "bailing_hybrid",
    "hidden_size": 1536,
    "intermediate_size": 4608,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "head_dim": 128,
    "vocab_size": 157184,
    "max_position_embeddings": 131072,
    "rms_norm_eps": 1e-6,
    "rope_theta": 6000000,
    "layer_group_size": 4,
    "short_conv_kernel_size": 4,
    "no_kda_lora": True,
    "kda_safe_gate": True,
    "kda_lower_bound": -5,
    "q_lora_rank": 256,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "qk_head_dim": 192,
    "v_head_dim": 128,
    "rope_interleave": True,
    "gated_attention_proj_granularity_type": "head_wise",
    "num_experts": 128,
    "num_shared_experts": 1,
    "num_experts_per_tok": 8,
    "n_group": 8,
    "topk_group": 4,
    "moe_intermediate_size": 512,
    "moe_shared_expert_intermediate_size": 512,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "moe_router_enable_expert_bias": True,
    "tie_word_embeddings": False,
    "use_qkv_bias": False,
    "pad_token_id": 156892,
    "eos_token_id": 156895,
}
KAI_HINTS = {
    "quant_method": "kai",
    "kai_matmul_triplet": "f32_qai8dxp_qsi4c32p",
    "kai_matmul_layout": "mxk_nxk",
    "kai_matmul_tile_cfg": "qai8dxp1x8_qsi4c32p8x8_1x8x32",
}


def expected_shapes() -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {
        "model.word_embeddings.weight": [157184, 1536],
        "model.norm.weight": [1536],
        "lm_head.weight": [157184, 1536],
    }
    for layer in range(24):
        prefix = f"model.layers.{layer}"
        shapes[f"{prefix}.input_layernorm.weight"] = [1536]
        shapes[f"{prefix}.post_attention_layernorm.weight"] = [1536]
        attention = f"{prefix}.attention"
        if layer in FULL_ATTENTION_LAYERS:
            shapes.update(
                {
                    f"{attention}.q_a_proj.weight": [256, 1536],
                    f"{attention}.q_a_layernorm.weight": [256],
                    f"{attention}.q_b_proj.weight": [3072, 256],
                    f"{attention}.kv_a_proj_with_mqa.weight": [576, 1536],
                    f"{attention}.kv_a_layernorm.weight": [512],
                    f"{attention}.kv_b_proj.weight": [4096, 512],
                    f"{attention}.g_proj.weight": [16, 1536],
                    f"{attention}.dense.weight": [1536, 2048],
                }
            )
        else:
            shapes.update(
                {
                    f"{attention}.q_proj.weight": [2048, 1536],
                    f"{attention}.k_proj.weight": [2048, 1536],
                    f"{attention}.v_proj.weight": [2048, 1536],
                    f"{attention}.q_conv1d.weight": [2048, 1, 4],
                    f"{attention}.k_conv1d.weight": [2048, 1, 4],
                    f"{attention}.v_conv1d.weight": [2048, 1, 4],
                    f"{attention}.f_proj.weight": [2048, 1536],
                    f"{attention}.A_log": [16],
                    f"{attention}.dt_bias": [2048],
                    f"{attention}.b_proj.weight": [16, 1536],
                    f"{attention}.g_proj.weight": [2048, 1536],
                    f"{attention}.o_norm.weight": [128],
                    f"{attention}.o_proj.weight": [1536, 2048],
                }
            )
        mlp = f"{prefix}.mlp"
        if layer == 0:
            shapes[f"{mlp}.gate_proj.weight"] = [4608, 1536]
            shapes[f"{mlp}.up_proj.weight"] = [4608, 1536]
            shapes[f"{mlp}.down_proj.weight"] = [1536, 4608]
        else:
            shapes[f"{mlp}.gate.weight"] = [128, 1536]
            shapes[f"{mlp}.gate.expert_bias"] = [128]
            for expert in range(128):
                expert_prefix = f"{mlp}.experts.{expert}"
                shapes[f"{expert_prefix}.gate_proj.weight"] = [512, 1536]
                shapes[f"{expert_prefix}.up_proj.weight"] = [512, 1536]
                shapes[f"{expert_prefix}.down_proj.weight"] = [1536, 512]
            shapes[f"{mlp}.shared_experts.gate_proj.weight"] = [512, 1536]
            shapes[f"{mlp}.shared_experts.up_proj.weight"] = [512, 1536]
            shapes[f"{mlp}.shared_experts.down_proj.weight"] = [1536, 512]
    assert len(shapes) == 9283
    return shapes


def validate_config(checkpoint_config: dict, runtime_config: dict) -> None:
    mismatches = [
        f"{key}={checkpoint_config.get(key)!r}, expected {value!r}"
        for key, value in OFFICIAL_CONTRACT.items()
        if checkpoint_config.get(key) != value
    ]
    if mismatches:
        raise AssertionError("Checkpoint contract mismatch: " + "; ".join(mismatches))
    runtime_mismatches = [
        f"{key}={runtime_config.get(key)!r}, expected {value!r}"
        for key, value in OFFICIAL_CONTRACT.items()
        if runtime_config.get(key) != value
    ]
    if runtime_mismatches:
        raise AssertionError("Runtime contract mismatch: " + "; ".join(runtime_mismatches))
    if runtime_config.get("max_cache_length") != 2048:
        raise AssertionError("Mobile runtime must use the reviewed 2048-token cache bound")
    expected_impl = (
        "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_"
        "qai8dxp1x8_qsi4c32p8x8_1x8x32"
    )
    if runtime_config.get("linear_impl_type") != expected_impl:
        raise AssertionError("Runtime config does not select the reviewed KAI W4A32 path")


def validate_recipe(shapes: dict[str, list[int]], recipe: dict) -> set[str]:
    matched: set[str] = set()
    for pattern_text, entry in recipe.items():
        pattern = re.compile(pattern_text)
        names = {name for name in shapes if pattern.fullmatch(name)}
        if not names:
            raise AssertionError(f"Quantization pattern matched no tensors: {pattern_text}")
        overlap = matched & names
        if overlap:
            raise AssertionError(f"Quantization patterns overlap: {sorted(overlap)[:5]}")
        hints = entry.get("hints", {})
        for key, value in KAI_HINTS.items():
            if hints.get(key) != value:
                raise AssertionError(f"{pattern_text}: invalid {key}")
        if hints.get("replace") is not True:
            raise AssertionError(f"{pattern_text}: KAI weights must replace source weights")
        for name in names:
            if shapes[name] != hints.get("shape"):
                raise AssertionError(
                    f"{pattern_text}: {name} has shape {shapes[name]}, recipe says {hints.get('shape')}"
                )
        matched.update(names)

    expected_quantized = {
        name
        for name, shape in shapes.items()
        if name == "lm_head.weight"
        or (
            name.endswith(".weight")
            and len(shape) == 2
            and name != "model.word_embeddings.weight"
            and not name.endswith(".mlp.gate.weight")
        )
    }
    if matched != expected_quantized:
        raise AssertionError(
            "Quantized tensor coverage mismatch: "
            f"missing={sorted(expected_quantized - matched)[:8]}, "
            f"extra={sorted(matched - expected_quantized)[:8]}"
        )
    return matched


def read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as file:
        raw_length = file.read(8)
        if len(raw_length) != 8:
            raise AssertionError(f"Truncated safetensors header: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        raw_header = file.read(header_length)
    if len(raw_header) != header_length:
        raise AssertionError(f"Truncated safetensors metadata: {path}")
    return json.loads(raw_header)


def validate_shards(checkpoint: Path, weight_map: dict[str, str], shapes: dict[str, list[int]]) -> None:
    actual: dict[str, tuple[str, list[int]]] = {}
    for shard_name in sorted(set(weight_map.values())):
        header = read_safetensors_header(checkpoint / shard_name)
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            if name in actual:
                raise AssertionError(f"Duplicate tensor across shards: {name}")
            actual[name] = (descriptor["dtype"], descriptor["shape"])
    if set(actual) != set(shapes):
        raise AssertionError(
            f"Shard tensor set mismatch: missing={sorted(set(shapes) - set(actual))[:8]}, "
            f"extra={sorted(set(actual) - set(shapes))[:8]}"
        )
    for name, expected_shape in shapes.items():
        expected_dtype = (
            "F32"
            if name.endswith(".A_log")
            or name.endswith(".dt_bias")
            or name.endswith(".gate.expert_bias")
            else "BF16"
        )
        if actual[name] != (expected_dtype, expected_shape):
            raise AssertionError(f"{name}: expected {(expected_dtype, expected_shape)}, got {actual[name]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--runtime-config", type=Path, default=Path(__file__).with_name("config_tiny_w4a32_kai.json"))
    parser.add_argument("--quant-config", type=Path, default=Path(__file__).with_name("quant_cfg_tiny_w4a32_kai.json"))
    parser.add_argument("--observed-revision", required=True)
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if args.observed_revision != PINNED_REVISION:
        raise AssertionError(
            f"Checkpoint revision {args.observed_revision} does not match pinned {PINNED_REVISION}"
        )
    checkpoint_config = json.loads((args.checkpoint / "config.json").read_text())
    runtime_config = json.loads(args.runtime_config.read_text())
    recipe = json.loads(args.quant_config.read_text())
    validate_config(checkpoint_config, runtime_config)
    shapes = expected_shapes()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map", {})
    if set(weight_map) != set(shapes):
        raise AssertionError(
            f"Index tensor set mismatch: missing={sorted(set(shapes) - set(weight_map))[:8]}, "
            f"extra={sorted(set(weight_map) - set(shapes))[:8]}"
        )
    validate_recipe(shapes, recipe)
    if not args.index_only:
        validate_shards(args.checkpoint, weight_map, shapes)
    print(
        "LING3_CHECKPOINT_OK "
        f"model={PINNED_MODEL_ID} revision={PINNED_REVISION} tensors={len(shapes)} "
        f"shards={len(set(weight_map.values()))} mode={'index' if args.index_only else 'full'}"
    )


if __name__ == "__main__":
    main()
