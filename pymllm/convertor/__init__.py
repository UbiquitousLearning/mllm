# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from __future__ import annotations

import os
import json
import importlib
from .model_file_v2 import ModelFileV2
from ..ffi import MLLM_FIND_NUMPY_AVAILABLE, MLLM_FIND_TORCH_AVAILABLE
from typing import Dict

if MLLM_FIND_NUMPY_AVAILABLE:
    import numpy as np
if MLLM_FIND_TORCH_AVAILABLE:
    import torch
MLLM_FIND_SAFETENSORS_AVAILABLE = importlib.util.find_spec("safetensors") is not None
if MLLM_FIND_SAFETENSORS_AVAILABLE:
    import safetensors
    from safetensors import safe_open


def load_model(file_path: str) -> Dict:
    """
    Load a model from file. Supports safetensors and torch formats.
    For safetensors, also supports model.index.json files.

    Args:
        file_path: Path to the model file

    Returns:
        Dictionary containing the model parameters
    """
    # Check if it's a safetensors file or index file
    if (
        file_path.endswith(".safetensors")
        or ".safetensors.index.json" in file_path
        or file_path.endswith(".index.json")
    ):
        if not MLLM_FIND_SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors package is not available")

        # Handle index files
        if file_path.endswith(".index.json") or ".safetensors.index.json" in file_path:
            with open(file_path, "r") as f:
                index_data = json.load(f)

            # Get directory of index file
            index_dir = os.path.dirname(file_path)

            # Load all tensors from shard files
            state_dict = {}
            weight_map = index_data.get("weight_map", {})

            # Group tensors by shard file
            shard_tensors = {}
            for tensor_name, shard_file in weight_map.items():
                if shard_file not in shard_tensors:
                    shard_tensors[shard_file] = []
                shard_tensors[shard_file].append(tensor_name)

            # Load tensors from each shard
            for shard_file, tensor_names in shard_tensors.items():
                shard_path = os.path.join(index_dir, shard_file)
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for tensor_name in tensor_names:
                        state_dict[tensor_name] = f.get_tensor(tensor_name)

            return state_dict
        else:
            # Single safetensors file
            state_dict = {}
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
            return state_dict

    # Handle torch files
    elif file_path.endswith((".pt", ".pth", ".bin")):
        if not MLLM_FIND_TORCH_AVAILABLE:
            raise ImportError("torch package is not available")

        return torch.load(file_path, map_location="cpu")

    else:
        raise ValueError(f"Unsupported file format for: {file_path}")
