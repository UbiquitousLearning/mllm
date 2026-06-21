"""Quantization settings for model weights and KV cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class QuantizationConfig:
    """Quantization configuration for weights and KV cache."""

    # Weight quantization method (e.g. "awq", "gptq", "fp8", None for no quant)
    method: Optional[str] = None
    # KV cache data type override
    kv_cache_dtype: Literal[
        "auto", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2"
    ] = "auto"
