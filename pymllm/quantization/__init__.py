"""Quantization infrastructure for pymllm."""

from pymllm.quantization.quant_config import (
    QuantizationConfig,
    get_quantization_config,
    list_quantization_methods,
    register_quantization,
)

# Import methods module to trigger @register_quantization decorators
import pymllm.quantization.methods  # noqa: F401

__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "list_quantization_methods",
    "register_quantization",
]
