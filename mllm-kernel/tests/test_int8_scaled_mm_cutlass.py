"""Correctness tests for CUTLASS int8_scaled_mm kernel."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch


def _cutlass_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "mllm_kernel"
        / "cuda"
        / "csrc"
        / "gemm"
        / "int8"
        / "int8_scaled_mm_cutlass.cu"
    ).read_text()


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


def test_cutlass_wrapper_rejects_unsupported_out_dtype(monkeypatch):
    from mllm_kernel.cuda.jit import int8_scaled_mm_cutlass as cutlass_wrapper

    class FakeModule:
        def int8_scaled_mm(self, *args, **kwargs):
            return torch.empty((1, 8), dtype=torch.bfloat16)

    monkeypatch.setattr(cutlass_wrapper, "_load_module", lambda: FakeModule())

    mat_a = torch.empty((1, 16), dtype=torch.int8)
    mat_b = torch.empty((16, 8), dtype=torch.int8)
    scales_a = torch.empty((1,), dtype=torch.float32)
    scales_b = torch.empty((8,), dtype=torch.float32)

    with pytest.raises(ValueError, match="out_dtype"):
        cutlass_wrapper.int8_scaled_mm(
            mat_a, mat_b, scales_a, scales_b, torch.float32,
        )


def test_cutlass_jit_uses_current_gpu_arch_for_compile(monkeypatch):
    import torch.utils.cpp_extension as cpp_extension

    from mllm_kernel.cuda.jit import int8_scaled_mm_cutlass as cutlass_wrapper

    calls = {}

    class FakeLoadedModule:
        pass

    def fake_load(**kwargs):
        calls.update(kwargs)
        return FakeLoadedModule()

    monkeypatch.setattr(cutlass_wrapper, "_module", None)
    monkeypatch.setattr(cutlass_wrapper, "_module_arch", None, raising=False)
    monkeypatch.setattr(cutlass_wrapper, "_CUTLASS_INC", None)
    monkeypatch.setattr(
        cutlass_wrapper,
        "_find_cutlass_include",
        lambda: "/tmp/cutlass/include",
    )
    monkeypatch.setattr(
        cutlass_wrapper.torch.cuda,
        "get_device_capability",
        lambda: (8, 9),
    )
    monkeypatch.setattr(cpp_extension, "load", fake_load)

    cutlass_wrapper._load_module()

    assert "-arch=sm_89" in calls["extra_cuda_cflags"]
    assert calls["name"].endswith("_sm_89")
    assert calls["build_directory"].endswith("sm_89")


def test_cutlass_dispatch_keeps_sglang_sm80_sm89_split():
    source = _cutlass_source()

    assert "if (sm_version == 86 || sm_version == 89)" in source
    assert "sm89_dispatch_shape<cutlass::bfloat16_t" in source
    assert "sm89_dispatch_shape<cutlass::half_t" in source
    assert "sm80_dispatch_shape<cutlass::bfloat16_t" in source
    assert "sm80_dispatch_shape<cutlass::half_t" in source


def test_sm80_dispatch_keeps_sglang_small_m_large_n_stage_split():
    source = _cutlass_source()

    assert source.count("if (n <= 4096)") >= 3


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
