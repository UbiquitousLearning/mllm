from __future__ import annotations

import pytest
import torch

from mllm_kernel.cuda.jit import int8_scaled_mm


def _reference_int8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
  out_i32 = torch.matmul(mat_a.to(torch.float32), mat_b.to(torch.float32))
  out = out_i32 * scales_a.view(-1, 1).to(torch.float32) * scales_b.view(1, -1).to(
      torch.float32
  )
  if bias is not None:
    out = out + bias.to(torch.float32)
  return out.to(out_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("M,N,K", [(1, 64, 32), (8, 128, 96), (32, 96, 128)])
def test_int8_scaled_mm_matches_reference(
    M: int,
    N: int,
    K: int,
    out_dtype: torch.dtype,
    with_bias: bool,
) -> None:
  torch.manual_seed(2026)
  mat_a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
  mat_b = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
  scales_a = torch.rand((M, 1), dtype=torch.float32, device="cuda") + 1e-4
  scales_b = torch.rand((N,), dtype=torch.float32, device="cuda") + 1e-4
  bias = (
      torch.randn((N,), dtype=out_dtype, device="cuda")
      if with_bias
      else None
  )

  out = int8_scaled_mm(
      mat_a,
      mat_b,
      scales_a,
      scales_b,
      out_dtype=out_dtype,
      bias=bias,
  )
  ref = _reference_int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)
  torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)
