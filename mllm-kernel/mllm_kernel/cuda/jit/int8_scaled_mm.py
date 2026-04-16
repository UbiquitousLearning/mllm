from __future__ import annotations

from typing import Optional

import torch

from mllm_kernel.jit_utils import cache_once, jit, make_cpp_args


@cache_once
def _make_int8_scaled_mm_kernel(out_dtype: torch.dtype):
  cpp_args = make_cpp_args(out_dtype)

  @jit(
      args=[out_dtype],
      device="cuda",
      cuda_files=["gemm/int8/int8_scaled_mm.cuh"],
      cpp_wrappers=[],
      cuda_wrappers=[("int8_scaled_mm", f"int8_scaled_mm<{cpp_args}>")],
      func_name="int8_scaled_mm",
  )
  def _kernel(
      compiled_module,
      mat_a: torch.Tensor,
      mat_b: torch.Tensor,
      scales_a: torch.Tensor,
      scales_b: torch.Tensor,
      bias: torch.Tensor,
      out: torch.Tensor,
  ) -> None:
    compiled_module.int8_scaled_mm(
        mat_a,
        mat_b,
        scales_a,
        scales_b,
        bias,
        out,
    )

  return _kernel


def int8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  if out_dtype not in (torch.float16, torch.bfloat16):
    raise ValueError(f"Unsupported out_dtype: {out_dtype}")

  if mat_a.dim() != 2 or mat_b.dim() != 2:
    raise ValueError("mat_a and mat_b must be 2D tensors")
  if mat_a.shape[1] != mat_b.shape[0]:
    raise ValueError(
        f"Incompatible shapes: mat_a={tuple(mat_a.shape)}, mat_b={tuple(mat_b.shape)}"
    )

  mat_a = mat_a.contiguous()
  mat_b = mat_b.contiguous()
  scales_a = scales_a.reshape(-1).contiguous().to(torch.float32)
  scales_b = scales_b.reshape(-1).contiguous().to(torch.float32)

  if bias is None:
    bias = torch.empty(0, device=mat_a.device, dtype=out_dtype)
  else:
    bias = bias.contiguous().to(out_dtype)

  out = torch.empty(
      (mat_a.shape[0], mat_b.shape[1]),
      device=mat_a.device,
      dtype=out_dtype,
  )
  kernel = _make_int8_scaled_mm_kernel(out_dtype)
  kernel(
      mat_a,
      mat_b,
      scales_a,
      scales_b,
      bias,
      out,
  )
  return out
