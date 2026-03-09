"""Gated RMSNorm layer for Qwen3.5 GDN attention.

Computes ``rmsnorm(x, weight, eps) * silu(z)`` using a fused CUDA kernel
from mllm-kernel.  Falls back to PyTorch when the kernel is unavailable.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.utils import set_weight_attrs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to load the mllm-kernel fused CUDA implementation
# ---------------------------------------------------------------------------
_HAS_MLLM_KERNEL_CUDA = False
try:
    from mllm_kernel.cuda.jit.rms_norm_gated import (
        rms_norm_gated as _mllm_rms_norm_gated,
    )

    _HAS_MLLM_KERNEL_CUDA = True
except Exception:
    _mllm_rms_norm_gated = None


# ---------------------------------------------------------------------------
# Pure-PyTorch fallback
# ---------------------------------------------------------------------------


def _rms_norm_gated_pytorch(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    norm_before_gate: bool = True,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation."""
    dtype = x.dtype
    x_fp32 = x.float()
    w_fp32 = weight.float()
    z_fp32 = z.float() if z is not None else None

    if z_fp32 is not None and not norm_before_gate:
        x_fp32 = x_fp32 * F.silu(z_fp32)

    variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    out = x_fp32 * rstd * w_fp32

    if z_fp32 is not None and norm_before_gate:
        out = out * F.silu(z_fp32)

    return out.to(dtype)


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------


def rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    norm_before_gate: bool = True,
) -> torch.Tensor:
    """Compute (optionally gated) RMS normalization.

    Uses the fused mllm-kernel CUDA implementation when available,
    otherwise falls back to a pure-PyTorch implementation.
    """
    if _HAS_MLLM_KERNEL_CUDA and x.is_cuda:
        return _mllm_rms_norm_gated(x, weight, z=z, eps=eps)
    return _rms_norm_gated_pytorch(
        x, weight, z=z, eps=eps, norm_before_gate=norm_before_gate,
    )


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------


class RMSNormGated(MllmBaseLayer):
    """Gated RMS Normalization layer for Qwen3.5 GDN attention.

    Computes::

        output = rmsnorm(x, weight) * silu(z)     # z is not None
        output = rmsnorm(x, weight)                # z is None

    Uses a fused CUDA kernel from mllm-kernel for maximum throughput.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input (and weight vector).
    eps : float
        Small constant for numerical stability.
    norm_before_gate : bool
        If ``True``  (default): ``rmsnorm(x) * silu(z)``.
        If ``False``:            ``rmsnorm(x * silu(z))``.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        group_size: Optional[int] = None,
        norm_before_gate: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm_before_gate = norm_before_gate

        factory_kwargs = {}
        if device is not None:
            factory_kwargs["device"] = device
        if dtype is not None:
            factory_kwargs["dtype"] = dtype

        self.weight = Parameter(torch.ones(hidden_size, **factory_kwargs))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return rms_norm_gated(
            x, self.weight, z=z, eps=self.eps,
            norm_before_gate=self.norm_before_gate,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, eps={self.eps}, "
            f"norm_before_gate={self.norm_before_gate}"
        )
