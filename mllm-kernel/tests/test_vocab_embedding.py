from __future__ import annotations

import pytest
import torch

from mllm_kernel.cuda.jit.vocab_embedding import (
    assemble_deepstack_embedding,
    embedding_lookup,
    embedding_lookup_multimodal,
    embedding_lookup_with_image,
)


def _make_lookup_inputs(
    *,
    num_tokens: int,
    vocab_size: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    """Build random (input_ids, embedding_table) for embedding_lookup.

    All token IDs are valid indices into embedding_table, so the result
    should be bit-exact against torch.index_select.
    """
    torch.manual_seed(seed)
    device = "cuda"
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), device=device, dtype=torch.int32
    )
    embedding_table = torch.randn(
        vocab_size, hidden_size, device=device, dtype=dtype
    )
    return input_ids, embedding_table


def _make_lookup_with_image_inputs(
    *,
    num_tokens: int,
    vocab_size: int,
    hidden_size: int,
    image_token_len: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    """Build mixed text/image inputs for embedding_lookup_with_image.

    The first half of input_ids are text tokens in [0, vocab_size),
    the second half are image tokens in [vocab_size, vocab_size + image_token_len).
    """
    torch.manual_seed(seed)
    device = "cuda"
    n_text = num_tokens // 2
    n_image = num_tokens - n_text
    text_ids = torch.randint(
        0, vocab_size, (n_text,), device=device, dtype=torch.int32
    )
    image_ids = torch.randint(
        vocab_size, vocab_size + image_token_len,
        (n_image,), device=device, dtype=torch.int32,
    )
    input_ids = torch.cat([text_ids, image_ids])
    embedding_table = torch.randn(
        vocab_size, hidden_size, device=device, dtype=dtype
    )
    image_embeds = torch.randn(
        image_token_len, hidden_size, device=device, dtype=dtype
    )
    return input_ids, embedding_table, image_embeds


def _make_deepstack_inputs(
    *,
    num_tokens: int,
    vocab_size: int,
    hidden_size: int,
    num_image_tokens: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    """Build image-only inputs for assemble_deepstack_embedding.

    All token IDs are in [vocab_size, vocab_size + num_image_tokens).
    """
    torch.manual_seed(seed)
    device = "cuda"
    input_ids = torch.randint(
        vocab_size, vocab_size + num_image_tokens,
        (num_tokens,), device=device, dtype=torch.int32,
    )
    deepstack_features = torch.randn(
        num_image_tokens, hidden_size, device=device, dtype=dtype
    )
    return input_ids, deepstack_features


def _make_multimodal_inputs(
    *,
    num_tokens: int,
    vocab_size: int,
    hidden_size: int,
    image_token_len: int,
    audio_token_len: int,
    image_token_id: int,
    audio_token_id: int,
    dtype: torch.dtype,
    seed: int = 0,
):
    """Build mixed text/image/audio inputs for embedding_lookup_multimodal.

    Roughly one third each of text, image, and audio tokens, shuffled.
    multimodal_indices are valid for each token type: image indices in
    [0, image_token_len), audio indices in [0, audio_token_len).
    """
    torch.manual_seed(seed)
    device = "cuda"

    n_text = num_tokens // 3
    n_image = num_tokens // 3
    n_audio = num_tokens - n_text - n_image

    text_ids = torch.randint(
        0, vocab_size, (n_text,), device=device, dtype=torch.int32
    )
    image_ids = torch.full(
        (n_image,), image_token_id, device=device, dtype=torch.int32
    )
    audio_ids = torch.full(
        (n_audio,), audio_token_id, device=device, dtype=torch.int32
    )
    input_ids = torch.cat([text_ids, image_ids, audio_ids])

    text_idx = torch.zeros(n_text, device=device, dtype=torch.int32)
    image_idx = torch.randint(
        0, image_token_len, (n_image,), device=device, dtype=torch.int32
    )
    audio_idx = torch.randint(
        0, audio_token_len, (n_audio,), device=device, dtype=torch.int32
    )
    multimodal_indices = torch.cat([text_idx, image_idx, audio_idx])

    perm = torch.randperm(num_tokens, device=device)
    input_ids = input_ids[perm]
    multimodal_indices = multimodal_indices[perm]

    embedding_table = torch.randn(
        vocab_size, hidden_size, device=device, dtype=dtype
    )
    image_embeds = torch.randn(
        image_token_len, hidden_size, device=device, dtype=dtype
    )
    audio_embeds = torch.randn(
        audio_token_len, hidden_size, device=device, dtype=dtype
    )
    return input_ids, embedding_table, multimodal_indices, image_embeds, audio_embeds


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_lookup_matches_torch(dtype: torch.dtype):
    """embedding_lookup must produce bit-exact results vs torch.index_select."""
    num_tokens = 257
    vocab_size = 32000
    hidden_size = 1024

    input_ids, embedding_table = _make_lookup_inputs(
        num_tokens=num_tokens,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        dtype=dtype,
        seed=2026,
    )

    output_ref = torch.index_select(embedding_table, 0, input_ids)

    output = embedding_lookup(input_ids, embedding_table)
    torch.cuda.synchronize()

    assert torch.equal(output, output_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_lookup_out_of_range_tokens():
    """Out-of-range token IDs (negative or >= vocab_size) must produce zero vectors."""
    vocab_size = 100
    hidden_size = 64
    dtype = torch.float16

    torch.manual_seed(2026)
    device = "cuda"
    embedding_table = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    input_ids = torch.tensor(
        [-1, vocab_size, vocab_size + 100], device=device, dtype=torch.int32
    )

    output = embedding_lookup(input_ids, embedding_table)
    torch.cuda.synchronize()

    expected = torch.zeros(3, hidden_size, dtype=dtype, device=device)
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_lookup_with_image_matches_reference(dtype: torch.dtype):
    """embedding_lookup_with_image must match a naive per-token reference.

    Text tokens [0, vocab_size) come from embedding_table; image tokens
    [vocab_size, vocab_size + image_token_len) come from image_embeds.
    """
    num_tokens = 257
    vocab_size = 1000
    hidden_size = 1024
    image_token_len = 576

    input_ids, embedding_table, image_embeds = _make_lookup_with_image_inputs(
        num_tokens=num_tokens,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        image_token_len=image_token_len,
        dtype=dtype,
        seed=2026,
    )

    input_ids_cpu = input_ids.cpu()
    embedding_table_cpu = embedding_table.cpu()
    image_embeds_cpu = image_embeds.cpu()
    ref = torch.zeros(num_tokens, hidden_size, dtype=dtype)
    for i in range(num_tokens):
        tid = input_ids_cpu[i].item()
        if tid >= vocab_size:
            ref[i] = image_embeds_cpu[tid - vocab_size]
        elif tid >= 0:
            ref[i] = embedding_table_cpu[tid]

    output = embedding_lookup_with_image(input_ids, embedding_table, image_embeds)
    torch.cuda.synchronize()

    assert torch.equal(output.cpu(), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_lookup_with_image_out_of_range():
    """Image token indices beyond image_token_len must produce zero vectors."""
    vocab_size = 100
    hidden_size = 64
    image_token_len = 10
    dtype = torch.float16

    torch.manual_seed(2026)
    device = "cuda"
    embedding_table = torch.randn(vocab_size, hidden_size, device=device, dtype=dtype)
    image_embeds = torch.randn(image_token_len, hidden_size, device=device, dtype=dtype)
    input_ids = torch.tensor(
        [vocab_size + image_token_len, vocab_size + image_token_len + 100],
        device=device, dtype=torch.int32,
    )

    output = embedding_lookup_with_image(input_ids, embedding_table, image_embeds)
    torch.cuda.synchronize()

    expected = torch.zeros(2, hidden_size, dtype=dtype, device=device)
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_assemble_deepstack_embedding_matches_reference(dtype: torch.dtype):
    """assemble_deepstack_embedding must extract image embeddings correctly.

    Token IDs >= vocab_size are looked up from deepstack_features at
    index (token_id - vocab_size). Token IDs < vocab_size produce zeros.
    """
    num_tokens = 257
    vocab_size = 1000
    hidden_size = 1024
    num_image_tokens = 576

    input_ids, deepstack_features = _make_deepstack_inputs(
        num_tokens=num_tokens,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_image_tokens=num_image_tokens,
        dtype=dtype,
        seed=2026,
    )

    input_ids_cpu = input_ids.cpu()
    features_cpu = deepstack_features.cpu()
    ref = torch.zeros(num_tokens, hidden_size, dtype=dtype)
    for i in range(num_tokens):
        tid = input_ids_cpu[i].item()
        if tid >= vocab_size:
            ref[i] = features_cpu[tid - vocab_size]

    output = assemble_deepstack_embedding(input_ids, deepstack_features, vocab_size)
    torch.cuda.synchronize()

    assert torch.equal(output.cpu(), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_assemble_deepstack_text_tokens_produce_zeros():
    """All-text input must produce an all-zero output."""
    num_tokens = 32
    vocab_size = 100
    hidden_size = 64
    num_image_tokens = 10
    dtype = torch.float16

    torch.manual_seed(2026)
    device = "cuda"
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), device=device, dtype=torch.int32
    )
    deepstack_features = torch.randn(
        num_image_tokens, hidden_size, device=device, dtype=dtype
    )

    output = assemble_deepstack_embedding(input_ids, deepstack_features, vocab_size)
    torch.cuda.synchronize()

    expected = torch.zeros(num_tokens, hidden_size, dtype=dtype, device=device)
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_embedding_lookup_multimodal_matches_reference(dtype: torch.dtype):
    """embedding_lookup_multimodal must match a naive per-token reference.

    Token IDs matching image_token_id use multimodal_indices into image_embeds.
    Token IDs matching audio_token_id use multimodal_indices into audio_embeds.
    Other valid token IDs come from embedding_table.
    """
    num_tokens = 257
    vocab_size = 1000
    hidden_size = 1024
    image_token_len = 50
    audio_token_len = 30
    image_token_id = 32000
    audio_token_id = 32001

    (
        input_ids,
        embedding_table,
        multimodal_indices,
        image_embeds,
        audio_embeds,
    ) = _make_multimodal_inputs(
        num_tokens=num_tokens,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        image_token_len=image_token_len,
        audio_token_len=audio_token_len,
        image_token_id=image_token_id,
        audio_token_id=audio_token_id,
        dtype=dtype,
        seed=2026,
    )

    input_ids_cpu = input_ids.cpu()
    mm_indices_cpu = multimodal_indices.cpu()
    embedding_table_cpu = embedding_table.cpu()
    image_embeds_cpu = image_embeds.cpu()
    audio_embeds_cpu = audio_embeds.cpu()
    ref = torch.zeros(num_tokens, hidden_size, dtype=dtype)
    for i in range(num_tokens):
        tid = input_ids_cpu[i].item()
        idx = mm_indices_cpu[i].item()
        if tid == image_token_id and 0 <= idx < image_token_len:
            ref[i] = image_embeds_cpu[idx]
        elif tid == audio_token_id and 0 <= idx < audio_token_len:
            ref[i] = audio_embeds_cpu[idx]
        elif 0 <= tid < vocab_size:
            ref[i] = embedding_table_cpu[tid]

    output = embedding_lookup_multimodal(
        input_ids, embedding_table, multimodal_indices,
        image_embeds, audio_embeds, image_token_id, audio_token_id,
    )
    torch.cuda.synchronize()

    assert torch.equal(output.cpu(), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_lookup_multimodal_none_embeds():
    """Passing None for both image and audio embeds must fall back to text-only lookup."""
    num_tokens = 64
    vocab_size = 1000
    hidden_size = 128
    image_token_id = 32000
    audio_token_id = 32001
    dtype = torch.float16

    torch.manual_seed(2026)
    device = "cuda"
    input_ids = torch.randint(
        0, vocab_size, (num_tokens,), device=device, dtype=torch.int32
    )
    embedding_table = torch.randn(
        vocab_size, hidden_size, device=device, dtype=dtype
    )
    multimodal_indices = torch.zeros(
        num_tokens, device=device, dtype=torch.int32
    )

    output = embedding_lookup_multimodal(
        input_ids, embedding_table, multimodal_indices,
        image_embeds=None, audio_embeds=None,
        image_token_id=image_token_id, audio_token_id=audio_token_id,
    )
    torch.cuda.synchronize()

    output_ref = torch.index_select(embedding_table, 0, input_ids)
    assert torch.equal(output, output_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_lookup_multimodal_image_only():
    """Multimodal lookup with only image embeds (audio_embeds=None)."""
    num_tokens = 64
    vocab_size = 1000
    hidden_size = 128
    image_token_len = 50
    image_token_id = 32000
    audio_token_id = 32001
    dtype = torch.float16

    torch.manual_seed(2026)
    device = "cuda"

    n_text = num_tokens // 2
    n_image = num_tokens - n_text
    text_ids = torch.randint(
        0, vocab_size, (n_text,), device=device, dtype=torch.int32
    )
    image_ids = torch.full(
        (n_image,), image_token_id, device=device, dtype=torch.int32
    )
    input_ids = torch.cat([text_ids, image_ids])

    text_idx = torch.zeros(n_text, device=device, dtype=torch.int32)
    image_idx = torch.randint(
        0, image_token_len, (n_image,), device=device, dtype=torch.int32
    )
    multimodal_indices = torch.cat([text_idx, image_idx])

    embedding_table = torch.randn(
        vocab_size, hidden_size, device=device, dtype=dtype
    )
    image_embeds = torch.randn(
        image_token_len, hidden_size, device=device, dtype=dtype
    )

    output = embedding_lookup_multimodal(
        input_ids, embedding_table, multimodal_indices,
        image_embeds=image_embeds, audio_embeds=None,
        image_token_id=image_token_id, audio_token_id=audio_token_id,
    )
    torch.cuda.synchronize()

    input_ids_cpu = input_ids.cpu()
    mm_indices_cpu = multimodal_indices.cpu()
    embedding_table_cpu = embedding_table.cpu()
    image_embeds_cpu = image_embeds.cpu()
    ref = torch.zeros(num_tokens, hidden_size, dtype=dtype)
    for i in range(num_tokens):
        tid = input_ids_cpu[i].item()
        idx = mm_indices_cpu[i].item()
        if tid == image_token_id and 0 <= idx < image_token_len:
            ref[i] = image_embeds_cpu[idx]
        elif 0 <= tid < vocab_size:
            ref[i] = embedding_table_cpu[tid]

    assert torch.equal(output.cpu(), ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_lookup_single_token():
    """Single token — exercises the minimal-work path."""
    dtype = torch.float16
    input_ids, embedding_table = _make_lookup_inputs(
        num_tokens=1, vocab_size=100, hidden_size=64, dtype=dtype, seed=2026,
    )

    output = embedding_lookup(input_ids, embedding_table)
    torch.cuda.synchronize()

    expected = embedding_table[input_ids[0].item()].unsqueeze(0)
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_lookup_all_same_token():
    """All tokens identical — verifies broadcast-like correctness."""
    num_tokens = 64
    vocab_size = 1000
    hidden_size = 128
    dtype = torch.float16

    torch.manual_seed(2026)
    device = "cuda"
    embedding_table = torch.randn(
        vocab_size, hidden_size, device=device, dtype=dtype
    )
    input_ids = torch.full(
        (num_tokens,), 42, device=device, dtype=torch.int32
    )

    output = embedding_lookup(input_ids, embedding_table)
    torch.cuda.synchronize()

    expected = embedding_table[42].unsqueeze(0).expand(num_tokens, -1)
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_lookup_large_hidden_size():
    """Large hidden_size (4096) — verifies vectorised copy handles long rows."""
    input_ids, embedding_table = _make_lookup_inputs(
        num_tokens=32, vocab_size=1000, hidden_size=4096,
        dtype=torch.float16, seed=2026,
    )

    output = embedding_lookup(input_ids, embedding_table)
    torch.cuda.synchronize()

    expected = torch.index_select(embedding_table, 0, input_ids)
    assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_embedding_lookup_large_batch():
    """Large batch (2048 tokens) — stress test for grid dimension."""
    input_ids, embedding_table = _make_lookup_inputs(
        num_tokens=2048, vocab_size=32000, hidden_size=256,
        dtype=torch.float16, seed=2026,
    )

    output = embedding_lookup(input_ids, embedding_table)
    torch.cuda.synchronize()

    expected = torch.index_select(embedding_table, 0, input_ids)
    assert torch.equal(output, expected)