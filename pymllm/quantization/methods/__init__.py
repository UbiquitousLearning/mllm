"""Quantization method implementations.

Importing this module triggers registration of all built-in quantization
methods via the ``@register_quantization`` decorator.
"""

from pymllm.quantization.methods.awq_marlin import (
    AWQMarlinConfig,
    AWQMarlinLinearMethod,
)
from pymllm.quantization.methods.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)

__all__ = [
    "AWQMarlinConfig",
    "AWQMarlinLinearMethod",
    "CompressedTensorsConfig",
    "CompressedTensorsLinearMethod",
]
