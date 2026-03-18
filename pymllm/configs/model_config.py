"""Lightweight model configuration: path + HuggingFace config handle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Minimal model config wrapping a HuggingFace PretrainedConfig.

    Attributes on ``hf_config`` are flattened onto this object::

        cfg = get_global_config().model
        cfg.hidden_size          # -> hf_config.hidden_size
        cfg.vocab_size           # -> hf_config.vocab_size
        cfg.text_config          # -> hf_config.text_config (multimodal)
    """

    # Populated at runtime via ``transformers.AutoConfig.from_pretrained``
    hf_config: Optional[Any] = field(default=None, repr=False)

    def __getattr__(self, name: str) -> Any:
        hf = object.__getattribute__(self, "hf_config")
        if hf is not None and hasattr(hf, name):
            return getattr(hf, name)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}' "
            f"(also not found on hf_config)"
        )
