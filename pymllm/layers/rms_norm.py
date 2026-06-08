from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import flashinfer
from torch.nn import Parameter

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.utils import set_weight_attrs


_PATCHED_CUDA_DEVICE_PROPERTIES = False


class _CudaDevicePropertiesProxy:
    def __init__(self, props):
        self._props = props

    def __getattr__(self, name: str):
        if name == "shared_memory_per_block_optin":
            return _infer_shared_memory_per_block_optin(self._props)
        return getattr(self._props, name)


def _infer_shared_memory_per_block_optin(props) -> int:
    """Infer opt-in shared memory for older PyTorch device properties."""
    if hasattr(props, "shared_memory_per_block_optin"):
        return int(props.shared_memory_per_block_optin)
    if hasattr(props, "shared_memory_per_multiprocessor"):
        return int(props.shared_memory_per_multiprocessor)
    return int(getattr(props, "shared_memory_per_block", 0))


def _patch_cuda_device_properties_for_flashinfer_norm() -> None:
    """Provide a missing PyTorch device property required by FlashInfer norm.

    Some Jetson PyTorch builds expose neither ``shared_memory_per_block_optin``
    nor ``shared_memory_per_multiprocessor`` on ``torch.cuda.DeviceProperties``.
    FlashInfer norm kernels query that attribute while choosing their CUTE
    kernel config.  Wrap the properties object so FlashInfer can still choose
    a valid shared-memory limit instead of falling back to slow PyTorch RMSNorm.
    """
    global _PATCHED_CUDA_DEVICE_PROPERTIES
    if _PATCHED_CUDA_DEVICE_PROPERTIES or not torch.cuda.is_available():
        return

    original_get_device_properties = torch.cuda.get_device_properties
    props = original_get_device_properties(0)
    if hasattr(props, "shared_memory_per_block_optin"):
        _PATCHED_CUDA_DEVICE_PROPERTIES = True
        return

    def patched_get_device_properties(*args, **kwargs):
        props = original_get_device_properties(*args, **kwargs)
        if hasattr(props, "shared_memory_per_block_optin"):
            return props
        return _CudaDevicePropertiesProxy(props)

    torch.cuda.get_device_properties = patched_get_device_properties
    _PATCHED_CUDA_DEVICE_PROPERTIES = True


def _torch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    x_norm = x_fp32 * torch.rsqrt(var + eps)
    return x_norm.to(dtype=x.dtype) * weight


class RMSNorm(MllmBaseLayer):
    """RMSNorm layer implemented with FlashInfer kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = Parameter(torch.empty(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim == hidden_size ({self.hidden_size}), "
                f"but got input shape {tuple(x.shape)}"
            )

        if residual is not None:
            try:
                _patch_cuda_device_properties_for_flashinfer_norm()
                flashinfer.norm.fused_add_rmsnorm(
                    x, residual, self.weight.data, self.eps
                )
                return x, residual
            except Exception:
                residual = x + residual
                return _torch_rmsnorm(residual, self.weight, self.eps), residual

        try:
            _patch_cuda_device_properties_for_flashinfer_norm()
            # FlashInfer rmsnorm accepts 2D/3D input; flatten higher-rank tensors to 2D.
            if x.dim() in (2, 3):
                return flashinfer.norm.rmsnorm(x, self.weight, self.eps)

            original_shape = x.shape
            x_2d = x.reshape(-1, self.hidden_size)
            out = flashinfer.norm.rmsnorm(x_2d, self.weight, self.eps)
            return out.reshape(original_shape)
        except Exception:
            return _torch_rmsnorm(x, self.weight, self.eps)


class GemmaRMSNorm(MllmBaseLayer):
    """Gemma-style RMSNorm layer implemented with FlashInfer kernel."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = Parameter(torch.empty(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if residual is not None:
            _patch_cuda_device_properties_for_flashinfer_norm()
            flashinfer.norm.gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.eps
            )
            return x, residual

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected last dim == hidden_size ({self.hidden_size}), "
                f"but got input shape {tuple(x.shape)}"
            )

        # gemma_rmsnorm is defined on 2D input; flatten other ranks to 2D.
        _patch_cuda_device_properties_for_flashinfer_norm()
        if x.dim() == 2:
            return flashinfer.norm.gemma_rmsnorm(x, self.weight, self.eps)

        original_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        out = flashinfer.norm.gemma_rmsnorm(x_2d, self.weight, self.eps)
        return out.reshape(original_shape)
