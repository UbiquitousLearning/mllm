from __future__ import annotations

import pytest
import torch

from mllm_kernel.cuda.jit.create_kv_indices import create_kv_indices


def _make_batch(
    *,
    max_reqs: int,
    max_ctx: int,
    batch_size: int,
    use_start_offsets: bool,
    seed: int = 0,
):
    """Construct a random-but-bounded test batch for create_kv_indices.

    The constraints ensure that for every sequence i:
        0 <= kv_start_idx[i]
        0 < page_kernel_lens[i]
        kv_start_idx[i] + page_kernel_lens[i] <= max_ctx
    so the kernel never reads beyond the ReqToTokenPool row.
    """
    # Use a CUDA generator for randperm (which requires matching device)
    # and a separate CPU generator for randint (which only accepts CPU).
    g_cuda = torch.Generator(device="cuda").manual_seed(seed)
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)

    device = "cuda"
    # req_to_token[req_slot, position] -> kv_index (here we simply use a
    # monotonically increasing pattern so correctness is easy to check).
    req_to_token = torch.arange(
        max_reqs * max_ctx, dtype=torch.int32, device=device
    ).reshape(max_reqs, max_ctx)

    # Sample distinct request slots for the batch.
    assert batch_size <= max_reqs
    req_pool_indices = torch.randperm(max_reqs, generator=g_cuda, device=device)[
        :batch_size
    ].to(torch.int32)

    # For each sequence choose a valid (start, length) pair.
    page_kernel_lens_list = []
    kv_start_idx_list = []
    for _ in range(batch_size):
        # ensure at least 1 token per sequence
        L = int(torch.randint(1, max_ctx, (1,), generator=g_cpu).item())
        if use_start_offsets:
            start_max = max_ctx - L
            start = int(torch.randint(0, max(start_max, 1), (1,), generator=g_cpu).item())
        else:
            start = 0
        page_kernel_lens_list.append(L)
        kv_start_idx_list.append(start)

    page_kernel_lens = torch.tensor(
        page_kernel_lens_list, dtype=torch.int32, device=device
    )
    kv_start_idx = torch.tensor(kv_start_idx_list, dtype=torch.int32, device=device)

    # Build kv_indptr prefix sums.
    kv_indptr = torch.empty(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[0] = 0
    kv_indptr[1:] = torch.cumsum(page_kernel_lens, dim=0)

    kv_indices = torch.empty(
        int(kv_indptr[-1].item()), dtype=torch.int32, device=device
    )

    return (
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_start_offsets", [False, True])
@pytest.mark.parametrize(
    "batch_size,max_reqs,max_ctx",
    [
        (1, 4, 16),        # minimal batch
        (4, 8, 64),        # small batch
        (32, 64, 512),     # medium batch, longer context
        (128, 256, 2048),  # larger batch, stress inner loop
    ],
)
def test_create_kv_indices_matches_reference(
    use_start_offsets: bool,
    batch_size: int,
    max_reqs: int,
    max_ctx: int,
):
    """create_kv_indices must match a naive PyTorch reference implementation.

    The reference is computed on CPU using explicit loops over
    (request_slot, start, length); the CUDA kernel must produce identical
    flat kv_indices for the same inputs.
    """
    (
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx,
        kv_indices,
    ) = _make_batch(
        max_reqs=max_reqs,
        max_ctx=max_ctx,
        batch_size=batch_size,
        use_start_offsets=use_start_offsets,
        seed=2026,
    )

    # Call CUDA kernel (kv_start_idx can be None to exercise that path).
    create_kv_indices(
        req_to_token,
        req_pool_indices,
        page_kernel_lens,
        kv_indptr,
        kv_start_idx if use_start_offsets else None,
        kv_indices,
    )
    torch.cuda.synchronize()

    # Naive reference on CPU.
    req_to_token_cpu = req_to_token.cpu()
    req_pool_indices_cpu = req_pool_indices.cpu().to(torch.long)
    page_kernel_lens_cpu = page_kernel_lens.cpu()
    kv_start_idx_cpu = kv_start_idx.cpu()

    ref_segments = []
    for i in range(batch_size):
        req = req_pool_indices_cpu[i].item()
        start = kv_start_idx_cpu[i].item() if use_start_offsets else 0
        L = page_kernel_lens_cpu[i].item()
        row = req_to_token_cpu[req, start : start + L]
        ref_segments.append(row)
    ref = torch.cat(ref_segments, dim=0)

    assert kv_indices.shape == ref.shape
    assert torch.equal(kv_indices.cpu(), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_single_token_per_sequence():
    """Each sequence has exactly 1 token — exercises the minimal-work path."""
    device = "cuda"
    bs = 8
    max_ctx = 32
    req_to_token = torch.arange(bs * max_ctx, dtype=torch.int32, device=device).reshape(bs, max_ctx)
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
    page_kernel_lens = torch.ones(bs, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(bs + 1, dtype=torch.int32, device=device)
    kv_indices = torch.empty(bs, dtype=torch.int32, device=device)

    create_kv_indices(req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, None, kv_indices)
    torch.cuda.synchronize()

    # Each sequence contributes req_to_token[i, 0].
    expected = req_to_token[:, 0]
    assert torch.equal(kv_indices, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_oversized_output_buffer():
    """kv_indices buffer is larger than needed (prefill path uses +256 padding)."""
    device = "cuda"
    bs = 4
    max_ctx = 64
    req_to_token = torch.arange(bs * max_ctx, dtype=torch.int32, device=device).reshape(bs, max_ctx)
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=device)
    page_kernel_lens = torch.full((bs,), 10, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, bs * 10 + 1, 10, dtype=torch.int32, device=device)
    # Allocate with extra padding, like the prefill path does.
    kv_indices = torch.full((bs * 10 + 256,), -1, dtype=torch.int32, device=device)

    create_kv_indices(req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, None, kv_indices)
    torch.cuda.synchronize()

    # First bs*10 entries should match; padding should remain -1.
    ref_segments = []
    for i in range(bs):
        ref_segments.append(req_to_token[i, :10])
    ref = torch.cat(ref_segments, dim=0)
    assert torch.equal(kv_indices[:bs * 10], ref)
    assert torch.all(kv_indices[bs * 10:] == -1)
