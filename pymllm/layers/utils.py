"""Utility functions for layers."""

from typing import Any, Dict

import torch


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Dict[str, Any] | None,
) -> None:
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor or parameter.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
            Common attributes include:
            - output_dim: The dimension along which to shard the weight (typically 0 for output dim)
            - input_dim: The input dimension (typically 1 for input dim)
            - weight_loader: A callable to load weights into this parameter
            - packed_dim: The dimension along which the weight is packed (for quantization)
            - packed_factor: The packing factor (for quantization)

    Example:
        >>> weight = nn.Parameter(torch.empty(100, 64))
        >>> set_weight_attrs(weight, {
        ...     "output_dim": 0,
        ...     "input_dim": 1,
        ...     "weight_loader": my_loader_func,
        ... })
    """
    if weight_attrs is None:
        return

    for key, value in weight_attrs.items():
        if hasattr(weight, key):
            raise AttributeError(
                f"Overwriting existing tensor attribute: {key}. "
                f"Existing value: {getattr(weight, key)}, "
                f"New value: {value}"
            )
        setattr(weight, key, value)
