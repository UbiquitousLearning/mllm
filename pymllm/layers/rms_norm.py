from __future__ import annotations

import torch
import flashinfer
from torch.nn import Parameter

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.utils import set_weight_attrs


class RMSNorm(MllmBaseLayer):
    """RMSNorm layer implemented with FlashInfer kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = Parameter(torch.empty(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim == hidden_size ({self.hidden_size}), "
                f"but got input shape {tuple(x.shape)}"
            )

        # FlashInfer rmsnorm accepts 2D/3D input; flatten higher-rank tensors to 2D.
        if x.dim() in (2, 3):
            return flashinfer.norm.rmsnorm(x, self.weight, self.eps)

        original_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        out = flashinfer.norm.rmsnorm(x_2d, self.weight, self.eps)
        return out.reshape(original_shape)


class GemmaRMSNorm(MllmBaseLayer):
    """Gemma-style RMSNorm layer implemented with FlashInfer kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = Parameter(torch.empty(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim == hidden_size ({self.hidden_size}), "
                f"but got input shape {tuple(x.shape)}"
            )

        # gemma_rmsnorm is defined on 2D input; flatten other ranks to 2D.
        if x.dim() == 2:
            return flashinfer.norm.gemma_rmsnorm(x, self.weight, self.eps)

        original_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        out = flashinfer.norm.gemma_rmsnorm(x_2d, self.weight, self.eps)
        return out.reshape(original_shape)
