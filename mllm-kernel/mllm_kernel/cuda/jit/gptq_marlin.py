"""GPTQ Marlin GEMM CUDA JIT kernel.

Performs quantized matrix multiplication using the Marlin kernel for
GPTQ/AWQ-style W4A16 or W8A16 quantized weights.

Usage::

    from mllm_kernel.cuda.jit.gptq_marlin import gptq_marlin_gemm

    output = gptq_marlin_gemm(
        a, c, b_q_weight, b_scales, global_scale, b_zeros,
        g_idx, perm, workspace, b_q_type_id,
        size_m, size_n, size_k,
    )
"""

from __future__ import annotations

from typing import Optional

import torch

from mllm_kernel.jit_utils import cache_once, jit, make_cpp_args

# Constants matching device::marlin:: in marlin.cuh
_MAX_THREAD_N = 256


@cache_once
def _make_gptq_marlin_gemm_kernel(dtype: torch.dtype):
    """JIT-compile the GPTQ Marlin GEMM kernel for a specific dtype."""
    cpp_args = make_cpp_args(dtype)

    @jit(
        args=[dtype],
        device="cuda",
        cuda_files=["gemm/marlin/gptq_marlin.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[("gptq_marlin_gemm", f"gptq_marlin_gemm<{cpp_args}>")],
        func_name="gptq_marlin_gemm",
    )
    def _kernel(
        compiled_module,
        a: torch.Tensor,
        b_q_weight: torch.Tensor,
        b_scales: torch.Tensor,
        global_scale: torch.Tensor,
        b_zeros: torch.Tensor,
        g_idx: torch.Tensor,
        perm: torch.Tensor,
        c: torch.Tensor,
        c_tmp: torch.Tensor,
        a_tmp: torch.Tensor,
        workspace: torch.Tensor,
        b_q_type_id: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
    ) -> None:
        compiled_module.gptq_marlin_gemm(
            a,
            b_q_weight,
            b_scales,
            global_scale,
            b_zeros,
            g_idx,
            perm,
            c,
            c_tmp,
            a_tmp,
            workspace,
            b_q_type_id,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
            is_zp_float,
        )

    return _kernel


def _or_empty(
    t: Optional[torch.Tensor], device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t if t is not None else torch.empty(0, device=device, dtype=dtype)


def gptq_marlin_gemm(
    a: torch.Tensor,
    c: Optional[torch.Tensor],
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    b_zeros: Optional[torch.Tensor],
    g_idx: Optional[torch.Tensor],
    perm: Optional[torch.Tensor],
    workspace: torch.Tensor,
    b_q_type_id: int,
    size_m: int,
    size_n: int,
    size_k: int,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool = False,
) -> torch.Tensor:
    """Perform quantized GEMM using the Marlin kernel.

    Parameters
    ----------
    a : torch.Tensor
        Input activation tensor, shape ``(size_m, size_k)``, fp16 or bf16.
    c : torch.Tensor or None
        Output buffer, shape ``(size_m, size_n)``. Allocated if ``None``.
    b_q_weight : torch.Tensor
        Quantized weight in Marlin layout, int32.
    b_scales : torch.Tensor
        Per-group quantization scales.
    global_scale : torch.Tensor or None
        Global scale for FP8 quantization.
    b_zeros : torch.Tensor or None
        Per-group zero points (for AWQ-style asymmetric quantization).
    g_idx : torch.Tensor or None
        Group indices for activation reordering.
    perm : torch.Tensor or None
        Permutation indices for activation reordering.
    workspace : torch.Tensor
        Workspace buffer for synchronization.
    b_q_type_id : int
        ScalarType id for the quantized weight type.
    size_m : int
        Batch dimension.
    size_n : int
        Output dimension.
    size_k : int
        Reduction dimension.
    is_k_full : bool
        Whether the full K dimension is present (no TP split on K).
    use_atomic_add : bool
        Use atomic add for output reduction.
    use_fp32_reduce : bool
        Use fp32 for global reduction.
    is_zp_float : bool
        Whether zero points are float16 type.

    Returns
    -------
    torch.Tensor
        Output tensor, shape ``(size_m, size_n)``.
    """
    device = a.device

    # Allocate output if not provided
    if c is None:
        c = torch.empty((size_m, size_n), dtype=a.dtype, device=device)

    # Early return for zero-size M
    if size_m == 0:
        return c

    # Determine activation ordering
    has_act_order = (
        g_idx is not None
        and perm is not None
        and g_idx.numel() > 0
        and perm.numel() > 0
    )

    # Allocate c_tmp for fp32 reduce
    if use_fp32_reduce:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_m_block = min(((size_m + 15) // 16) * 16, 64)
        c_tmp = torch.empty(
            sms * max_m_block * _MAX_THREAD_N,
            dtype=torch.float32,
            device=device,
        )
    else:
        c_tmp = torch.empty(0, dtype=torch.float32, device=device)

    # Allocate a_tmp for act_order column permutation
    if has_act_order:
        a_tmp = torch.empty((size_m, size_k), dtype=a.dtype, device=device)
    else:
        a_tmp = torch.empty(0, dtype=a.dtype, device=device)

    # Convert Optional tensors to empty tensors
    global_scale_t = _or_empty(global_scale, device, a.dtype)
    b_zeros_t = _or_empty(b_zeros, device, torch.int32)
    g_idx_t = _or_empty(g_idx, device, torch.int32)
    perm_t = _or_empty(perm, device, torch.int32)

    kernel = _make_gptq_marlin_gemm_kernel(a.dtype)
    kernel(
        a,
        b_q_weight,
        b_scales,
        global_scale_t,
        b_zeros_t,
        g_idx_t,
        perm_t,
        c,
        c_tmp,
        a_tmp,
        workspace,
        b_q_type_id,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )

    return c
