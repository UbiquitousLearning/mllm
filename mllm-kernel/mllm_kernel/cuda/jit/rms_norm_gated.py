"""Fused RMSNorm + SiLU gating CUDA JIT kernel for Qwen3.5 GDN attention.

Computes ``rmsnorm(x, weight, eps) * silu(z)`` in a single fused pass.

Usage::

    from mllm_kernel.cuda.jit.rms_norm_gated import rms_norm_gated

    output = rms_norm_gated(x, weight, z=gate, eps=1e-6)
"""

from __future__ import annotations

import torch

from mllm_kernel.jit_utils import cache_once, jit


@cache_once
def _make_rms_norm_gated_kernel():
    """JIT-compile the fused RMSNorm+gating CUDA kernel."""

    @jit(
        args=[],
        device="cuda",
        cuda_files=["rms_norm_gated.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[
            ("rms_norm_gated", "RMSNormGatedKernel::run"),
        ],
        func_name="rms_norm_gated",
    )
    def _kernel(
        compiled_module,
        output: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
    ) -> None:
        compiled_module.rms_norm_gated(output, input, weight, gate, eps)

    return _kernel


def rms_norm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm with optional SiLU gating.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor, shape ``(M, N)`` or ``(..., N)``.
    weight : torch.Tensor
        Normalization weight, shape ``(N,)``.
    z : torch.Tensor or None
        Optional gating tensor, same shape as ``x``.
        If provided: ``output = rmsnorm(x) * silu(z)``
    eps : float
        Epsilon for numerical stability.

    Returns
    -------
    torch.Tensor
        Output with same shape and dtype as ``x``.
    """
    x_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    if z is not None:
        z_2d = z.reshape(-1, z.shape[-1])
        if z_2d.stride(-1) != 1:
            z_2d = z_2d.contiguous()
    else:
        z_2d = x.new_empty(0)  # empty tensor signals "no gate" to the kernel

    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()

    output = torch.empty_like(x_2d)
    kernel = _make_rms_norm_gated_kernel()
    kernel(output, x_2d, weight.contiguous(), z_2d, eps)
    return output.reshape(x_shape)
