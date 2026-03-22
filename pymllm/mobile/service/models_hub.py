# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import os
import tvm_ffi
from typing import Tuple

from ..ffi import Session
from modelscope.hub.snapshot_download import snapshot_download


def session_qwen3(fp: str) -> Session:
    return tvm_ffi.get_global_func("mllm.service.session.qwen3")(fp)


MODEL_HUB_LOOKUP_TABLE = {
    "Qwen3-0.6B-w4a32kai": session_qwen3,
}


def create_session(model_name: str) -> Tuple[str, Session]:
    model_path = get_download_model_path(model_name)
    if model_path is None:
        model_path = download_mllm_model(model_name)
    last = os.path.basename(os.path.normpath(model_path))
    return last, MODEL_HUB_LOOKUP_TABLE[last](model_path)


# Down load model from model hub. ModelScope / HuggingFace, etc.
def download_mllm_model(model_name: str, model_dir: str = None) -> str:
    """
    Download a model from the model hub.

    Args:
        model_name: Name of the model to download
        model_dir: Directory to save the model

    Returns:
        Path to the downloaded model

    Raises:
        ValueError: If the model name is not supported
    """
    # Check if model is supported
    if model_name not in MODEL_HUB_LOOKUP_TABLE:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Supported models: {list(MODEL_HUB_LOOKUP_TABLE.keys())}"
        )

    try:
        # Download model from ModelScope
        model_path = snapshot_download("mllmTeam/" + model_name, local_dir=model_dir)
        return model_path
    except ImportError:
        # If modelscope is not installed, provide instructions
        raise ImportError(
            "modelscope is not installed. Please install it with: pip install modelscope"
        )
    except Exception as e:
        # Handle other exceptions during download
        raise RuntimeError(f"Failed to download model '{model_name}': {str(e)}")


def get_download_model_path(model_name_or_model_path: str):
    # If it's a path, check if it exists
    if os.path.exists(model_name_or_model_path):
        return model_name_or_model_path

    # Check if it's a supported model name
    if model_name_or_model_path not in MODEL_HUB_LOOKUP_TABLE:
        return None

    # Check ModelScope cache directory
    # Default cache directory is ~/.cache/modelscope/hub
    cache_dir = os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.expanduser("~/.cache/modelscope/hub/models/mllmTeam"),
    )
    if not cache_dir.endswith("mllmTeam"):
        model_cache_path = os.path.join(cache_dir, "mllmTeam")
    model_cache_path = os.path.join(cache_dir, model_name_or_model_path)

    # Check if model exists in cache
    if os.path.exists(model_cache_path):
        return model_cache_path

    # Model not found
    return None
