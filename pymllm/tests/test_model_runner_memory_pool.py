from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pymllm.executor import model_runner
from pymllm.executor.model_runner import ModelRunner


def _make_runner(
    *,
    mem_fraction_static: float,
    max_total_tokens: int | None = None,
    free_gb: float = 5.0,
    total_gb: float = 10.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        server_config=SimpleNamespace(
            mem_fraction_static=mem_fraction_static,
            max_total_tokens=max_total_tokens,
        ),
        device="cuda",
        gpu_id=0,
        kv_cache_dtype=torch.float16,
        num_kv_heads=1,
        head_dim=1,
        v_head_dim=1,
        num_hidden_layers=1,
        num_gdn_layers=0,
        _pre_model_load_available_gb=total_gb,
        _current_available_gb=free_gb,
    )


@pytest.fixture
def _cuda_available(monkeypatch):
    monkeypatch.setattr(model_runner.torch.cuda, "is_available", lambda: True)


def test_profile_max_num_tokens_treats_mem_fraction_static_as_static_pool_fraction(
    _cuda_available,
    monkeypatch,
):
    runner = _make_runner(mem_fraction_static=0.4, free_gb=5.0, total_gb=10.0)
    monkeypatch.setattr(
        model_runner,
        "get_available_gpu_memory",
        lambda *args, **kwargs: runner._current_available_gb,
    )

    # SGLang-compatible formula:
    # free_after_model - pre_model_free * (1 - mem_fraction_static)
    # = 5 GiB - 10 GiB * 0.6 = -1 GiB, so the pool is not allocatable.
    assert ModelRunner._profile_max_num_tokens(runner) <= 0


def test_profile_max_num_tokens_clamps_negative_capacity_and_records_diagnostics(
    _cuda_available,
    monkeypatch,
):
    runner = _make_runner(
        mem_fraction_static=0.4,
        max_total_tokens=4096,
        free_gb=5.0,
        total_gb=10.0,
    )
    monkeypatch.setattr(
        model_runner,
        "get_available_gpu_memory",
        lambda *args, **kwargs: runner._current_available_gb,
    )

    assert ModelRunner._profile_max_num_tokens(runner) == 0
    profile = runner._last_memory_profile
    assert profile.static_kv_budget_gb == pytest.approx(-1.0)
    assert profile.profiled_max_tokens == 0
    assert profile.requested_max_total_tokens == 4096


def test_kv_cache_memory_error_includes_profile_details_and_server_hint():
    runner = _make_runner(
        mem_fraction_static=0.75,
        max_total_tokens=4096,
        free_gb=1.99,
        total_gb=8.42,
    )
    runner._last_memory_profile = model_runner.MemoryProfileResult(
        pre_model_available_gb=8.42,
        available_gb=1.99,
        mem_fraction=0.75,
        static_kv_budget_gb=-0.11,
        cell_size_bytes=114688,
        profiled_max_tokens=0,
        requested_max_total_tokens=4096,
        effective_max_tokens=0,
        gdn_pool_gb=0.0,
    )

    message = ModelRunner._format_kv_cache_memory_error(runner)

    assert "profiled max_tokens=0" in message
    assert "requested max_total_tokens=4096" in message
    assert "static_kv_budget=-0.11 GB" in message
    assert "mem_fraction_static=0.75" in message
    assert "server mode may need a higher mem_fraction_static than bench_one_batch" in message
    assert "--server.mem_fraction_static" in message
    assert "--server.max_total_tokens" in message


def test_profile_max_num_tokens_caps_user_max_total_tokens_to_profiled_capacity(
    _cuda_available,
    monkeypatch,
):
    runner = _make_runner(
        mem_fraction_static=0.8,
        max_total_tokens=8_000_000_000,
        free_gb=5.0,
        total_gb=10.0,
    )
    monkeypatch.setattr(
        model_runner,
        "get_available_gpu_memory",
        lambda *args, **kwargs: runner._current_available_gb,
    )

    # Profiled capacity is 3 GiB / 4 bytes per token. A larger user limit should
    # not bypass profiling or force an oversized KV pool allocation.
    assert ModelRunner._profile_max_num_tokens(runner) == 805_306_368


def test_profile_max_num_tokens_uses_user_limit_when_below_profiled_capacity(
    _cuda_available,
    monkeypatch,
):
    runner = _make_runner(
        mem_fraction_static=0.8,
        max_total_tokens=4096,
        free_gb=5.0,
        total_gb=10.0,
    )
    monkeypatch.setattr(
        model_runner,
        "get_available_gpu_memory",
        lambda *args, **kwargs: runner._current_available_gb,
    )

    assert ModelRunner._profile_max_num_tokens(runner) == 4096


def test_available_gpu_memory_uses_system_memory_for_integrated_gpu(monkeypatch):
    monkeypatch.setattr(model_runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(model_runner.torch.cuda, "set_device", lambda gpu_id: None)
    monkeypatch.setattr(
        model_runner.torch.cuda,
        "get_device_properties",
        lambda gpu_id: SimpleNamespace(is_integrated=True),
    )
    monkeypatch.setattr(model_runner, "_get_system_available_memory_gb", lambda: 7.5)

    def fail_mem_get_info(*args, **kwargs):
        raise AssertionError("integrated GPU memory should use system available memory")

    monkeypatch.setattr(model_runner.torch.cuda, "mem_get_info", fail_mem_get_info)

    assert model_runner.get_available_gpu_memory("cuda", 0) == 7.5
