"""CUTLASS-based INT8 scaled matmul for SM80+ (Ampere).

JIT-compiled via torch.utils.cpp_extension.load on first use.
Compiled module is cached at ~/.cache/mllm_kernel/cutlass_int8_scaled_mm/.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

_module = None
_CSRC_DIR = Path(__file__).resolve().parent.parent / "csrc"
_CUTLASS_INC = None


def _find_cutlass_include() -> str:
    """Find CUTLASS include path."""
    # Check environment variable
    env_path = os.environ.get("CUTLASS_HOME")
    if env_path and os.path.isdir(os.path.join(env_path, "include", "cutlass")):
        return os.path.join(env_path, "include")

    # Check flashinfer bundled copy
    try:
        import flashinfer
        fi_path = os.path.join(
            os.path.dirname(flashinfer.__file__),
            "data", "cutlass", "include",
        )
        if os.path.isdir(os.path.join(fi_path, "cutlass")):
            return fi_path
    except ImportError:
        pass

    # Check common system paths
    for p in [
        "/usr/local/include",
        "/usr/include",
        "/usr/local/cuda/include",
    ]:
        if os.path.isdir(os.path.join(p, "cutlass")):
            return p

    raise RuntimeError(
        "CUTLASS include directory not found. Set CUTLASS_HOME or install "
        "flashinfer (which bundles CUTLASS headers)."
    )


def _load_module():
    global _module, _CUTLASS_INC
    if _module is not None:
        return _module

    from torch.utils.cpp_extension import load

    _CUTLASS_INC = _find_cutlass_include()

    cache_dir = os.path.expanduser("~/.cache/mllm_kernel/cutlass_int8_scaled_mm")
    os.makedirs(cache_dir, exist_ok=True)

    source = str(_CSRC_DIR / "gemm" / "int8" / "int8_scaled_mm_cutlass.cu")

    _module = load(
        name="mllm_cutlass_int8_scaled_mm",
        sources=[source],
        extra_include_paths=[
            _CUTLASS_INC,
            str(_CSRC_DIR),
        ],
        extra_cuda_cflags=[
            "-arch=sm_87",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--expt-relaxed-constexpr",
            "-std=c++17",
            "-diag-suppress=20013",
            "-diag-suppress=20015",
            "-O3",
        ],
        build_directory=cache_dir,
        verbose=False,
    )
    return _module


def int8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CUTLASS INT8 scaled matmul: out = (mat_a @ mat_b) * scales_a * scales_b + bias.

    Args:
        mat_a: [M, K] int8, row-major (contiguous)
        mat_b: [K, N] int8, column-major (stride(0)==1)
        scales_a: [M] float32, per-row scale for activations
        scales_b: [N] float32, per-column scale for weights
        out_dtype: torch.float16 or torch.bfloat16
        bias: optional [N] tensor, same dtype as out_dtype

    Returns:
        [M, N] tensor of out_dtype
    """
    mod = _load_module()

    # scales_a from Triton quant is (M,1) float32 — flatten to (M,)
    if scales_a.dim() == 2:
        scales_a = scales_a.squeeze(-1)

    dtype_str = "float16" if out_dtype == torch.float16 else "bfloat16"

    return mod.int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, dtype_str, bias)
