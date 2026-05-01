"""Model registry for pymllm.

Maps HuggingFace ``config.architectures[0]`` strings to pymllm model classes.
Models are imported lazily via ``importlib`` so that heavy dependencies (torch,
numpy, etc.) are only loaded when a model is actually requested.
"""

from __future__ import annotations

import importlib
import logging
from typing import Dict, Optional, Tuple, Type

import torch.nn as nn

logger = logging.getLogger(__name__)

# (module_path, class_name)
_MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "Qwen3ForCausalLM": (
        "pymllm.models.qwen3",
        "Qwen3ForCausalLM",
    ),
    "Qwen3VLForConditionalGeneration": (
        "pymllm.models.qwen3_vl",
        "Qwen3VLForConditionalGeneration",
    ),
    # Qwen3.5 (hybrid attention: full + GDN linear)
    "Qwen3_5ForCausalLM": (
        "pymllm.models.qwen3_5",
        "Qwen3_5ForCausalLM",
    ),
    "Qwen3_5ForConditionalGeneration": (
        "pymllm.models.qwen3_5",
        "Qwen3_5ForConditionalGeneration",
    ),
    "Gemma3nForCausalLM": (
        "pymllm.models.gemma3n",
        "Gemma3nForCausalLM",
    ),
    "Gemma3nForConditionalGeneration": (
        "pymllm.models.gemma3n",
        "Gemma3nForConditionalGeneration",
    ),
}


def get_model_class(architecture: str) -> Optional[Type[nn.Module]]:
    """Look up a pymllm model class by HuggingFace architecture string.

    Returns ``None`` if the architecture is not registered or cannot be
    imported.  The caller is responsible for raising an appropriate error.
    """
    entry = _MODEL_REGISTRY.get(architecture)
    if entry is None:
        return None

    module_path, class_name = entry
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        logger.info(
            "Resolved architecture %r -> %s.%s", architecture, module_path, class_name
        )
        return cls
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "Failed to import %s.%s for architecture %r: %s",
            module_path,
            class_name,
            architecture,
            exc,
        )
        return None
