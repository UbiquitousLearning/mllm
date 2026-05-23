from __future__ import annotations

import pytest
import torch

from pymllm.layers.rope import apply_mrope, apply_mrope_fused_


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_apply_mrope_fused_matches_reference_and_keeps_inputs_in_place():
    torch.manual_seed(0)
    num_tokens = 17
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 8
    mrope_section = [2, 1, 1]

    q = torch.randn(
        (num_tokens, num_q_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn(
        (num_tokens, num_kv_heads, head_dim),
        device="cuda",
        dtype=torch.float16,
    )
    positions = torch.randint(
        0,
        32,
        (3, num_tokens),
        device="cuda",
        dtype=torch.long,
    )
    cos_sin_cache = torch.randn(
        (32, head_dim),
        device="cuda",
        dtype=torch.float16,
    )

    expected_q, expected_k = apply_mrope(
        q,
        k,
        positions,
        cos_sin_cache,
        mrope_section,
        mrope_interleaved=True,
    )

    q_actual = q.clone()
    k_actual = k.clone()
    q_ptr = q_actual.data_ptr()
    k_ptr = k_actual.data_ptr()

    out_q, out_k = apply_mrope_fused_(
        q_actual,
        k_actual,
        positions,
        cos_sin_cache,
        mrope_section,
        mrope_interleaved=True,
    )
    torch.cuda.synchronize()

    assert out_q.data_ptr() == q_ptr
    assert out_k.data_ptr() == k_ptr
    assert q_actual.data_ptr() == q_ptr
    assert k_actual.data_ptr() == k_ptr
    torch.testing.assert_close(q_actual, expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_actual, expected_k, atol=1e-2, rtol=1e-2)
