"""Fused GDN decode CUDA JIT kernel.

Performs a single-token GDN (Gated Delta Net) recurrent update per request,
fusing gating + L2 normalization + delta rule + output computation into
one kernel.  Works on SM80+ (Ampere, Jetson Orin, Hopper, ...).

Usage::

    from mllm_kernel.cuda.jit.gdn_decode import gdn_decode

    output = gdn_decode(q, k, v, a, b, A_log, dt_bias, state_pool, cache_indices)
"""

from __future__ import annotations

import torch

from mllm_kernel.jit_utils import cache_once, jit


@cache_once
def _make_gdn_decode_kernel():
    """JIT-compile the fused GDN decode CUDA kernel."""

    @jit(
        args=[],
        device="cuda",
        cuda_files=["gdn_decode.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[
            ("gdn_decode", "GDNDecodeKernel::run"),
        ],
        func_name="gdn_decode",
    )
    def _kernel(
        compiled_module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_pool: torch.Tensor,
        cache_indices: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        compiled_module.gdn_decode(
            q, k, v, a, b, A_log, dt_bias, state_pool, cache_indices, output
        )

    return _kernel


def gdn_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """Fused GDN decode: gating + L2 norm + delta rule + output.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor, shape ``(bs, num_k_heads, head_k_dim)``, bf16/fp16.
    k : torch.Tensor
        Key tensor, shape ``(bs, num_k_heads, head_k_dim)``, bf16/fp16.
    v : torch.Tensor
        Value tensor, shape ``(bs, num_v_heads, head_v_dim)``, bf16/fp16.
    a : torch.Tensor
        Decay gate input, shape ``(bs, num_v_heads)``, bf16/fp16.
    b : torch.Tensor
        Update gate input, shape ``(bs, num_v_heads)``, bf16/fp16.
    A_log : torch.Tensor
        Log-space decay parameter, shape ``(num_v_heads,)``, float32.
    dt_bias : torch.Tensor
        Bias for decay gate, shape ``(num_v_heads,)``, float32.
    state_pool : torch.Tensor
        Pooled recurrent state, shape ``(pool_size, num_v_heads, head_v_dim, head_k_dim)``,
        float32.  Modified in-place.
    cache_indices : torch.Tensor
        Pool indices per request, shape ``(bs,)``, int64.

    Returns
    -------
    torch.Tensor
        Output tensor, shape ``(bs, num_v_heads, head_v_dim)``, same dtype as v.
    """
    bs = q.shape[0]
    num_v_heads = v.shape[1]
    head_v_dim = v.shape[2]

    output = torch.empty(bs, num_v_heads, head_v_dim, dtype=v.dtype, device=v.device)

    kernel = _make_gdn_decode_kernel()
    kernel(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        A_log.contiguous(),
        dt_bias.contiguous(),
        state_pool,
        cache_indices.to(torch.int64).contiguous(),
        output,
    )
    return output
