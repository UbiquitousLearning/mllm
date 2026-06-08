from __future__ import annotations

import torch

import pymllm.layers.rms_norm as rms_norm_module
from pymllm.layers.rms_norm import RMSNorm


def test_rms_norm_residual_fallback_returns_updated_residual(monkeypatch):
    def fail_fused_add_rmsnorm(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("force torch fallback")

    monkeypatch.setattr(
        rms_norm_module.flashinfer.norm,
        "fused_add_rmsnorm",
        fail_fused_add_rmsnorm,
    )

    norm = RMSNorm(hidden_size=3, eps=1e-6)
    norm.weight.data.fill_(1.0)
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    residual = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)

    _, residual_out = norm(x, residual)

    torch.testing.assert_close(residual_out, x + residual)


def test_flashinfer_norm_device_properties_patch_adds_optin(monkeypatch):
    class FakeProps:
        shared_memory_per_block = 49152
        shared_memory_per_multiprocessor = 167936

    monkeypatch.setattr(rms_norm_module, "_PATCHED_CUDA_DEVICE_PROPERTIES", False)
    monkeypatch.setattr(rms_norm_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        rms_norm_module.torch.cuda,
        "get_device_properties",
        lambda device=0: FakeProps(),
    )

    rms_norm_module._patch_cuda_device_properties_for_flashinfer_norm()

    props = rms_norm_module.torch.cuda.get_device_properties(0)
    assert props.shared_memory_per_block_optin == 167936
    assert props.shared_memory_per_block == 49152
