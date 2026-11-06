# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from __future__ import annotations

import os
import json
import importlib
from .model_file_v2 import ModelFileV2 as ModelFileV2
from ..ffi import MLLM_FIND_TORCH_AVAILABLE
from pathlib import Path
from typing import Dict, Sequence

if MLLM_FIND_TORCH_AVAILABLE:
    import torch
MLLM_FIND_SAFETENSORS_AVAILABLE = importlib.util.find_spec("safetensors") is not None
if MLLM_FIND_SAFETENSORS_AVAILABLE:
    from safetensors import safe_open


def _resolve_shard_path(index_dir: str, shard_file: str) -> str:
    """Resolve a shard name from a safetensors index inside its directory.

    Shard names come from the index file, so an absolute path or a ``..``
    segment must not let them point outside the directory that holds the
    index. The check is lexical on purpose: shards that are symlinks (as in a
    Hugging Face cache snapshot) keep working.

    Raises:
        ValueError: If the shard name escapes ``index_dir``.
    """
    if not isinstance(shard_file, str) or not shard_file:
        raise ValueError(f"Invalid shard file name in index: {shard_file!r}")
    normalized = os.path.normpath(shard_file)
    if (
        os.path.isabs(shard_file)
        or os.path.splitdrive(shard_file)[0]
        or normalized == os.pardir
        or normalized.startswith(os.pardir + os.sep)
    ):
        raise ValueError(
            f"Shard file {shard_file!r} in index points outside the model "
            f"directory {index_dir!r}"
        )
    return os.path.join(index_dir, normalized)


def load_model(file_path: str, include_prefixes: Sequence[str] = ()) -> Dict:
    """
    Load a model from file. Supports safetensors and torch formats.
    For safetensors, also supports model.index.json files.

    Args:
        file_path: Path to the model file or a directory containing a safetensors model
        include_prefixes: Optional parameter-name prefixes to load. Filtering an
            indexed model before opening its shards avoids loading unrelated
            vision or MTP tensors.

    Returns:
        Dictionary containing the model parameters
    """
    file_path = os.fspath(file_path)
    if isinstance(include_prefixes, (str, bytes)):
        raise TypeError("include_prefixes must be a sequence of non-empty strings")
    include_prefixes = tuple(include_prefixes)
    if any(not isinstance(prefix, str) or not prefix for prefix in include_prefixes):
        raise ValueError("include_prefixes must contain only non-empty strings")

    if os.path.isdir(file_path):
        model_dir = Path(file_path)
        index_files = sorted(model_dir.glob("*.safetensors.index.json"))
        if index_files:
            preferred_index = model_dir / "model.safetensors.index.json"
            if preferred_index in index_files:
                file_path = str(preferred_index)
            elif len(index_files) == 1:
                file_path = str(index_files[0])
            else:
                candidates = ", ".join(path.name for path in index_files)
                raise ValueError(
                    "Multiple safetensors indexes found without "
                    f"model.safetensors.index.json: {candidates}"
                )
        else:
            safetensors_files = sorted(model_dir.glob("*.safetensors"))
            if len(safetensors_files) != 1:
                raise ValueError(
                    f"Expected one safetensors file or an index in directory: {model_dir}"
                )
            file_path = str(safetensors_files[0])

    def should_load(name: str) -> bool:
        return not include_prefixes or any(
            name.startswith(prefix) for prefix in include_prefixes
        )

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
                if not should_load(tensor_name):
                    continue
                if shard_file not in shard_tensors:
                    shard_tensors[shard_file] = []
                shard_tensors[shard_file].append(tensor_name)

            # Load tensors from each shard
            for shard_file, tensor_names in shard_tensors.items():
                shard_path = _resolve_shard_path(index_dir, shard_file)
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for tensor_name in tensor_names:
                        state_dict[tensor_name] = f.get_tensor(tensor_name)

            return state_dict
        else:
            # Single safetensors file
            state_dict = {}
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    if should_load(k):
                        state_dict[k] = f.get_tensor(k)
            return state_dict

    # Handle torch files
    elif file_path.endswith((".pt", ".pth", ".bin")):
        if not MLLM_FIND_TORCH_AVAILABLE:
            raise ImportError("torch package is not available")

        state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
        if include_prefixes:
            state_dict = {
                name: tensor for name, tensor in state_dict.items() if should_load(name)
            }
        return state_dict

    else:
        raise ValueError(f"Unsupported file format for: {file_path}")
