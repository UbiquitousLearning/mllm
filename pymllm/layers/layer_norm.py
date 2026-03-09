from __future__ import annotations

import torch
import flashinfer
from torch.nn import Parameter

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.utils import set_weight_attrs


class LayerNorm(MllmBaseLayer):
    """LayerNorm layer implemented with FlashInfer kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        # flashinfer.norm.layernorm expects gamma/beta in fp32.
        self.weight = Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.bias = Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})
        set_weight_attrs(self.bias, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim == hidden_size ({self.hidden_size}), "
                f"but got input shape {tuple(x.shape)}"
            )
        if x.dtype != torch.bfloat16:
            raise TypeError(
                "flashinfer.norm.layernorm requires bfloat16 input, "
                f"but got {x.dtype}"
            )

        if x.dim() == 2:
            return flashinfer.norm.layernorm(x, self.weight, self.bias, self.eps)

        original_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        out = flashinfer.norm.layernorm(x_2d, self.weight, self.bias, self.eps)
        return out.reshape(original_shape)
