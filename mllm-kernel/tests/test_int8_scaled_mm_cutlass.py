"""Correctness tests for CUTLASS int8_scaled_mm kernel."""
from __future__ import annotations

import pytest
import torch


def _reference_int8_scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """fp32 reference implementation."""
    out = torch.matmul(mat_a.to(torch.float32), mat_b.to(torch.float32))
    out = out * scales_a.view(-1, 1).float() * scales_b.view(1, -1).float()
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


@pytest.fixture(scope="module")
def cutlass_module():
    """Load CUTLASS module once for all tests."""
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from mllm_kernel.cuda.jit.int8_scaled_mm_cutlass import int8_scaled_mm
    return int8_scaled_mm


@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize(
    "M,N,K",
    [
        (1, 64, 32),
        (1, 2048, 2048),
        (8, 128, 64),
        (16, 6144, 2048),
        (32, 2048, 2048),
        (93, 6144, 2048),
        (128, 2048, 6144),
    ],
)
def test_cutlass_matches_reference(
    cutlass_module, M, N, K, out_dtype, with_bias,
):
    torch.manual_seed(42)
    mat_a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
    mat_b = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
    # Make col-major B
    mat_b_col = mat_b.t().contiguous().t()

    scales_a = (torch.rand(M, dtype=torch.float32, device="cuda") + 0.01) * 0.01
    scales_b = (torch.rand(N, dtype=torch.float32, device="cuda") + 0.01) * 0.01
    bias = torch.randn(N, dtype=out_dtype, device="cuda") * 0.01 if with_bias else None

    out = cutlass_module(mat_a, mat_b_col, scales_a, scales_b, out_dtype, bias)
    ref = _reference_int8_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype, bias)

    torch.testing.assert_close(out, ref, atol=0.1, rtol=0.05)
