from __future__ import annotations

import pytest
import torch

from mllm_kernel.cuda.jit import can_use_store_cache, store_cache


def _make_inputs(
    *,
    batch_size: int,
    num_slots: int,
    row_dim: int,
    dtype: torch.dtype,
    index_dtype: torch.dtype,
    seed: int = 0,
):
    torch.manual_seed(seed)
    device = "cuda"
    k = torch.randn(batch_size, row_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, row_dim, device=device, dtype=dtype)
    # Use unique indices to avoid write conflicts on the same cache slot.
    indices = torch.randperm(num_slots, device=device)[:batch_size].to(index_dtype)
    k_cache = torch.zeros(num_slots, row_dim, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    return k, v, k_cache, v_cache, indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_store_cache_matches_torch_index(dtype: torch.dtype, index_dtype: torch.dtype):
    batch_size = 257
    num_slots = 4096
    row_dim = 8 * 128  # 1024 -> fp16 row_bytes=2048
    row_bytes = row_dim * torch.tensor([], dtype=dtype).element_size()

    assert can_use_store_cache(row_bytes), f"store_cache unavailable for row_bytes={row_bytes}"

    k, v, k_cache, v_cache, indices = _make_inputs(
        batch_size=batch_size,
        num_slots=num_slots,
        row_dim=row_dim,
        dtype=dtype,
        index_dtype=index_dtype,
        seed=2026,
    )

    k_ref = k_cache.clone()
    v_ref = v_cache.clone()
    k_ref[indices] = k
    v_ref[indices] = v

    store_cache(k, v, k_cache, v_cache, indices)
    torch.cuda.synchronize()

    assert torch.equal(k_cache, k_ref)
    assert torch.equal(v_cache, v_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_can_use_store_cache_rejects_invalid_row_bytes():
    assert not can_use_store_cache(2)
    assert not can_use_store_cache(6)
    assert can_use_store_cache(4)

