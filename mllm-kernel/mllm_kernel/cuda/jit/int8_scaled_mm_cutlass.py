"""CUTLASS-based INT8 scaled matmul for SM80+ (Ampere).

JIT-compiled via torch.utils.cpp_extension.load on first use.
Compiled module is cached per GPU arch at
~/.cache/mllm_kernel/cutlass_int8_scaled_mm/sm_XX/.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

_module = None
_module_arch = None
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


def _current_cuda_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}"
    if major != 8:
        raise RuntimeError(
            f"CUTLASS int8_scaled_mm supports SM80-SM89, got {arch}"
        )
    return arch


def _load_module():
    global _module, _module_arch, _CUTLASS_INC

    cuda_arch = _current_cuda_arch()
    if _module is not None and _module_arch == cuda_arch:
        return _module

    from torch.utils.cpp_extension import load

    _CUTLASS_INC = _find_cutlass_include()

    cache_dir = os.path.expanduser(
        os.path.join("~/.cache/mllm_kernel/cutlass_int8_scaled_mm", cuda_arch)
    )
    os.makedirs(cache_dir, exist_ok=True)

    source = str(_CSRC_DIR / "gemm" / "int8" / "int8_scaled_mm_cutlass.cu")

    _module = load(
        name=f"mllm_cutlass_int8_scaled_mm_{cuda_arch}",
        sources=[source],
        extra_include_paths=[
            _CUTLASS_INC,
            str(_CSRC_DIR),
        ],
        extra_cuda_cflags=[
            f"-arch={cuda_arch}",
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
    _module_arch = cuda_arch
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
    if out_dtype == torch.float16:
        dtype_str = "float16"
    elif out_dtype == torch.bfloat16:
        dtype_str = "bfloat16"
    else:
        raise ValueError(
            f"out_dtype must be torch.float16 or torch.bfloat16, got {out_dtype}"
        )

    mod = _load_module()

    # scales_a from Triton quant is (M,1) float32 — flatten to (M,)
    if scales_a.dim() == 2:
        scales_a = scales_a.squeeze(-1)

    return mod.int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, dtype_str, bias)
