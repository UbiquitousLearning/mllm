#!/usr/bin/env python3
"""
Qwen3.5 QNN AOT Weight Pre-baking Script (Phase 0)

This script prepares Qwen3.5 weights for the QNN AOT pipeline:
1. Pre-bakes GemmaRMSNorm weights: adds +1.0 to all RMSNorm weights that use
   add_unit_offset=true, so the QNN runtime can use standard RMSNorm (no offset).
2. Pre-computes partial RoPE sin/cos embedding tables for rotary_dim=64,
   rope_theta=10M, stored as model.mllm_max_sin_embedding / model.mllm_max_cos_embedding.
3. Writes output as mllm v2 weight file (.mllm).

Usage:
    python convert_weights.py \
        --input_path models/Qwen3.5-0.8B/model.safetensors.index.json \
        --output_path models/Qwen3.5-0.8B/qwen3_5_qnn.mllm \
        --max_position 2048

    Or from an existing mllm file (reads safetensors only):
    python convert_weights.py \
        --input_path models/Qwen3.5-0.8B \
        --output_path models/Qwen3.5-0.8B/qwen3_5_qnn.mllm
"""

import argparse
import json
import math
import os
import sys

import torch

# Add project root to path for pymllm imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pymllm.mobile.convertor import load_model
from pymllm.mobile.convertor.model_file_v2 import ModelFileV2


# Qwen3.5 0.8B config constants
NUM_HIDDEN_LAYERS = 24
FULL_ATTENTION_INTERVAL = 4
FULL_ATTENTION_INDICES = {3, 7, 11, 15, 19, 23}
HEAD_DIM = 256
PARTIAL_ROTARY_FACTOR = 0.25
ROTARY_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64
ROPE_THETA = 10_000_000.0
RMS_NORM_EPS = 1e-6


def is_gemma_rmsnorm_weight(name: str) -> bool:
    """Check if a weight name corresponds to a GemmaRMSNorm (add_unit_offset=true).

    These are:
    - All 24 layers: input_layernorm.weight, post_attention_layernorm.weight
    - Full attention layers: self_attn.q_norm.weight, self_attn.k_norm.weight
    - Final norm: model.language_model.norm.weight

    NOT included (standard RMSNorm, add_unit_offset=false):
    - GDN layers: linear_attn.norm.weight
    """
    if name == "model.language_model.norm.weight":
        return True
    if name.endswith(".input_layernorm.weight"):
        return True
    if name.endswith(".post_attention_layernorm.weight"):
        return True
    if name.endswith(".self_attn.q_norm.weight"):
        return True
    if name.endswith(".self_attn.k_norm.weight"):
        return True
    return False


def prebake_rmsnorm_weights(state_dict: dict) -> int:
    """Add +1.0 to all GemmaRMSNorm weights in-place. Returns count of modified tensors."""
    count = 0
    for name in list(state_dict.keys()):
        if is_gemma_rmsnorm_weight(name):
            state_dict[name] = state_dict[name].float() + 1.0
            count += 1
    return count


def compute_rope_tables(max_position: int, rotary_dim: int, rope_theta: float):
    """Compute sin/cos embedding tables for partial RoPE.

    Returns:
        sin_table: [1, max_position, rotary_dim] float32
        cos_table: [1, max_position, rotary_dim] float32
    """
    # inv_freq: [rotary_dim // 2]
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )

    # positions: [max_position]
    positions = torch.arange(max_position, dtype=torch.float32)

    # freqs: [max_position, rotary_dim // 2]
    freqs = torch.outer(positions, inv_freq)

    # emb: [max_position, rotary_dim] (repeat for pairs)
    emb = torch.cat([freqs, freqs], dim=-1)

    # [1, max_position, rotary_dim]
    cos_table = emb.cos().unsqueeze(0)
    sin_table = emb.sin().unsqueeze(0)

    return sin_table, cos_table


def find_safetensors_input(input_path: str) -> str:
    """Resolve input path to a loadable safetensors path."""
    if os.path.isdir(input_path):
        # Look for index file first
        index_path = os.path.join(input_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            return index_path
        # Look for single safetensors file
        for f in os.listdir(input_path):
            if f.endswith(".safetensors"):
                return os.path.join(input_path, f)
        raise FileNotFoundError(f"No safetensors files found in {input_path}")
    return input_path


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 QNN AOT Weight Pre-baking")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to HF model directory or safetensors index/file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output mllm v2 weight file path",
    )
    parser.add_argument(
        "--max_position",
        type=int,
        default=2048,
        help="Max position for RoPE table (default: 2048, matches max_cache_length)",
    )
    parser.add_argument(
        "--text_only",
        action="store_true",
        help="Only export text (language_model) weights, skip visual/mtp weights",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about each weight",
    )
    args = parser.parse_args()

    # 1. Load weights
    input_path = find_safetensors_input(args.input_path)
    print(f"Loading weights from: {input_path}")
    state_dict = load_model(input_path)
    print(f"Loaded {len(state_dict)} tensors")

    # Filter to text-only if requested
    if args.text_only:
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if k.startswith("model.language_model.") or k == "lm_head.weight"
        }
        print(f"Filtered to {len(state_dict)} text-only tensors")

    # 2. Pre-bake GemmaRMSNorm weights (+1.0)
    num_modified = prebake_rmsnorm_weights(state_dict)
    print(f"Pre-baked {num_modified} GemmaRMSNorm weights (+1.0 offset)")

    if args.verbose:
        for name in sorted(state_dict.keys()):
            if is_gemma_rmsnorm_weight(name):
                t = state_dict[name]
                print(f"  {name}: min={t.min():.4f} max={t.max():.4f} (after +1.0)")

    # 3. Pre-compute partial RoPE sin/cos tables
    sin_table, cos_table = compute_rope_tables(
        args.max_position, ROTARY_DIM, ROPE_THETA
    )
    state_dict["model.mllm_max_sin_embedding"] = sin_table
    state_dict["model.mllm_max_cos_embedding"] = cos_table
    print(
        f"Pre-computed RoPE tables: shape={list(sin_table.shape)}, "
        f"rotary_dim={ROTARY_DIM}, theta={ROPE_THETA:.0f}, max_pos={args.max_position}"
    )

    # 4. Cast remaining bf16 weights to fp32 for mllm v2 compatibility
    cast_count = 0
    for name in list(state_dict.keys()):
        if state_dict[name].dtype == torch.bfloat16:
            state_dict[name] = state_dict[name].float()
            cast_count += 1
    if cast_count > 0:
        print(f"Cast {cast_count} bf16 tensors to fp32")

    # 5. Write mllm v2 file
    print(f"Writing {len(state_dict)} tensors to: {args.output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    writer = ModelFileV2(
        args.output_path,
        "qwen3_5_qnn",
        "Streaming",
        max_params_descriptor_buffer_num=len(state_dict) + 16,
    )
    for name, tensor in state_dict.items():
        writer.streaming_write(name, tensor)
        if args.verbose:
            print(f"  Wrote {name}: {list(tensor.shape)} {tensor.dtype}")
    writer.finalize()

    file_size = os.path.getsize(args.output_path)
    print(f"Done. Output size: {file_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
