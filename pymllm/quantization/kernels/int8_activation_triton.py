"""Per-token INT8 activation quantization using Triton.

Ported from sglang int8_kernel.py (per_token_quant_int8).
Original: sglang/srt/layers/quantization/int8_kernel.py:28-89
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    """Triton kernel: per-token dynamic INT8 quantization.

    Each program instance handles one row (token).
    Computes absmax, derives scale, quantizes to int8.
    """
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def per_token_quant_int8(
    x: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic INT8 quantization.

    Args:
        x: Input tensor, any shape with last dim = hidden_dim. Must be contiguous.
        scale_dtype: Dtype for scale output (default float32).

    Returns:
        x_q: INT8 quantized tensor, same shape as x.
        scales: Per-token scales, shape = x.shape[:-1] + (1,).
    """
    assert x.is_contiguous(), "Input must be contiguous"

    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(
        x.shape[:-1] + (1,), device=x.device, dtype=scale_dtype
    )

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, scales
