# Copyright (c) MLLM Team.
# Licensed under the MIT License.
# Embedding kernels migrated from TensorRT-Edge-LLM.
# Reference: https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/main/cpp/kernels/embeddingKernels

from __future__ import annotations

from typing import Optional

import torch

from mllm_kernel.jit_utils import jit


# ============================================================================
# Op 1: embedding_lookup
# ============================================================================


@jit(
    device="cuda",
    cuda_files=["vocab_embedding.cuh"],
    cpp_wrappers=[],
    cuda_wrappers=[("embedding_lookup", "embedding_lookup")],
    func_name="embedding_lookup",
)
def _embedding_lookup_kernel(
    compiled_module, output: torch.Tensor, input_ids: torch.Tensor, embedding_table: torch.Tensor
) -> None:
    compiled_module.embedding_lookup(output, input_ids, embedding_table)


def embedding_lookup(input_ids: torch.Tensor, embedding_table: torch.Tensor) -> torch.Tensor:
    """
    Standard embedding lookup using vectorized CUDA kernel.

    Maps each token ID in input_ids to its corresponding row in embedding_table.
    Out-of-range token IDs produce zero vectors.

    Args:
        input_ids: Token IDs, shape [num_tokens], dtype int32, device cuda.
        embedding_table: Embedding weight matrix, shape [vocab_size, hidden_size],
                         dtype float16 or bfloat16, device cuda.

    Returns:
        Embedded output, shape [num_tokens, hidden_size], dtype matching embedding_table.

    Example:
        >>> import torch
        >>> from mllm_kernel.cuda.jit.vocab_embedding import embedding_lookup
        >>> ids = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
        >>> table = torch.randn(100, 256, dtype=torch.float16, device="cuda")
        >>> out = embedding_lookup(ids, table)
        >>> assert out.shape == (3, 256)
    """
    if input_ids.dtype != torch.int32:
        input_ids = input_ids.to(torch.int32)
    if not input_ids.is_contiguous():
        input_ids = input_ids.contiguous()
    if not embedding_table.is_contiguous():
        embedding_table = embedding_table.contiguous()

    num_tokens = input_ids.shape[0]
    hidden_size = embedding_table.shape[1]
    output = torch.empty(num_tokens, hidden_size, dtype=embedding_table.dtype, device=input_ids.device)

    _embedding_lookup_kernel(output, input_ids, embedding_table)
    return output


# ============================================================================
# Op 2: embedding_lookup_with_image
# ============================================================================


@jit(
    device="cuda",
    cuda_files=["vocab_embedding.cuh"],
    cpp_wrappers=[],
    cuda_wrappers=[("embedding_lookup_with_image", "embedding_lookup_with_image")],
    func_name="embedding_lookup_with_image",
)
def _embedding_lookup_with_image_kernel(
    compiled_module,
    output: torch.Tensor,
    input_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    image_embeds: torch.Tensor,
) -> None:
    compiled_module.embedding_lookup_with_image(output, input_ids, embedding_table, image_embeds)


def embedding_lookup_with_image(
    input_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    image_embeds: torch.Tensor,
) -> torch.Tensor:
    """
    Embedding lookup with image embedding insertion.

    For token IDs in [0, vocab_size): lookup from embedding_table.
    For token IDs >= vocab_size: lookup from image_embeds at index (token_id - vocab_size).

    Args:
        input_ids: Token IDs, shape [num_tokens], dtype int32, device cuda.
        embedding_table: Text embedding table, shape [vocab_size, hidden_size],
                         dtype float16 or bfloat16, device cuda.
        image_embeds: Image embeddings, shape [image_token_len, hidden_size],
                      dtype matching embedding_table, device cuda.

    Returns:
        Embedded output, shape [num_tokens, hidden_size], dtype matching embedding_table.

    Example:
        >>> import torch
        >>> from mllm_kernel.cuda.jit.vocab_embedding import embedding_lookup_with_image
        >>> ids = torch.tensor([0, 100, 101], dtype=torch.int32, device="cuda")
        >>> table = torch.randn(100, 256, dtype=torch.float16, device="cuda")
        >>> img = torch.randn(10, 256, dtype=torch.float16, device="cuda")
        >>> out = embedding_lookup_with_image(ids, table, img)
        >>> assert out.shape == (3, 256)
    """
    if input_ids.dtype != torch.int32:
        input_ids = input_ids.to(torch.int32)
    if not input_ids.is_contiguous():
        input_ids = input_ids.contiguous()
    if not embedding_table.is_contiguous():
        embedding_table = embedding_table.contiguous()
    if not image_embeds.is_contiguous():
        image_embeds = image_embeds.contiguous()

    num_tokens = input_ids.shape[0]
    hidden_size = embedding_table.shape[1]
    output = torch.empty(num_tokens, hidden_size, dtype=embedding_table.dtype, device=input_ids.device)

    _embedding_lookup_with_image_kernel(output, input_ids, embedding_table, image_embeds)
    return output


# ============================================================================
# Op 3: assemble_deepstack_embedding
# ============================================================================


@jit(
    device="cuda",
    cuda_files=["vocab_embedding.cuh"],
    cpp_wrappers=[],
    cuda_wrappers=[("assemble_deepstack_embedding", "assemble_deepstack_embedding")],
    func_name="assemble_deepstack_embedding",
)
def _assemble_deepstack_embedding_kernel(
    compiled_module,
    output: torch.Tensor,
    input_ids: torch.Tensor,
    deepstack_features: torch.Tensor,
    vocab_size: int,
) -> None:
    compiled_module.assemble_deepstack_embedding(output, input_ids, deepstack_features, vocab_size)


def assemble_deepstack_embedding(
    input_ids: torch.Tensor,
    deepstack_features: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """
    Extract image-only embeddings from deepstack features.

    Token IDs >= vocab_size: lookup from deepstack_features at index (token_id - vocab_size).
    Token IDs < vocab_size: zero output (text tokens handled elsewhere).

    Args:
        input_ids: Token IDs, shape [num_tokens], dtype int32, device cuda.
        deepstack_features: Deepstack feature embeddings, shape [num_image_tokens, hidden_size],
                            dtype float16 or bfloat16, device cuda.
        vocab_size: Vocabulary size (threshold for image token detection).

    Returns:
        Embedded output, shape [num_tokens, hidden_size], dtype matching deepstack_features.

    Example:
        >>> import torch
        >>> from mllm_kernel.cuda.jit.vocab_embedding import assemble_deepstack_embedding
        >>> ids = torch.tensor([0, 100, 101], dtype=torch.int32, device="cuda")
        >>> features = torch.randn(10, 256, dtype=torch.float16, device="cuda")
        >>> out = assemble_deepstack_embedding(ids, features, vocab_size=100)
        >>> assert out.shape == (3, 256)
    """
    if input_ids.dtype != torch.int32:
        input_ids = input_ids.to(torch.int32)
    if not input_ids.is_contiguous():
        input_ids = input_ids.contiguous()
    if not deepstack_features.is_contiguous():
        deepstack_features = deepstack_features.contiguous()

    num_tokens = input_ids.shape[0]
    hidden_size = deepstack_features.shape[1]
    output = torch.empty(num_tokens, hidden_size, dtype=deepstack_features.dtype, device=input_ids.device)

    _assemble_deepstack_embedding_kernel(output, input_ids, deepstack_features, vocab_size)
    return output


# ============================================================================
# Op 4: embedding_lookup_multimodal
# ============================================================================


@jit(
    device="cuda",
    cuda_files=["vocab_embedding.cuh"],
    cpp_wrappers=[],
    cuda_wrappers=[("embedding_lookup_multimodal", "embedding_lookup_multimodal")],
    func_name="embedding_lookup_multimodal",
)
def _embedding_lookup_multimodal_kernel(
    compiled_module,
    output: torch.Tensor,
    input_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    multimodal_indices: torch.Tensor,
    image_embeds: torch.Tensor,
    audio_embeds: torch.Tensor,
    image_token_id: int,
    audio_token_id: int,
) -> None:
    compiled_module.embedding_lookup_multimodal(
        output, input_ids, embedding_table, multimodal_indices,
        image_embeds, audio_embeds, image_token_id, audio_token_id
    )


def embedding_lookup_multimodal(
    input_ids: torch.Tensor,
    embedding_table: torch.Tensor,
    multimodal_indices: torch.Tensor,
    image_embeds: Optional[torch.Tensor] = None,
    audio_embeds: Optional[torch.Tensor] = None,
    image_token_id: int = -1,
    audio_token_id: int = -1,
) -> torch.Tensor:
    """
    Multimodal embedding lookup supporting text, image, and audio tokens.

    Uses multimodal_indices to determine the embedding index for image/audio tokens.
    Token IDs matching image_token_id or audio_token_id are looked up from their
    respective embedding tables. Other token IDs are looked up from embedding_table.

    Args:
        input_ids: Token IDs, shape [num_tokens], dtype int32, device cuda.
        embedding_table: Text embedding table, shape [vocab_size, hidden_size],
                         dtype float16 or bfloat16, device cuda.
        multimodal_indices: Indices for image/audio embeddings, shape [num_tokens],
                            dtype int32, device cuda.
        image_embeds: Image embeddings, shape [image_token_len, hidden_size],
                      dtype matching embedding_table, device cuda. Optional.
        audio_embeds: Audio embeddings, shape [audio_token_len, hidden_size],
                      dtype matching embedding_table, device cuda. Optional.
        image_token_id: Special token ID for image tokens. Default -1 (disabled).
        audio_token_id: Special token ID for audio tokens. Default -1 (disabled).

    Returns:
        Embedded output, shape [num_tokens, hidden_size], dtype matching embedding_table.

    Example:
        >>> import torch
        >>> from mllm_kernel.cuda.jit.vocab_embedding import embedding_lookup_multimodal
        >>> ids = torch.tensor([0, 32000, 32001, 1], dtype=torch.int32, device="cuda")
        >>> table = torch.randn(32000, 256, dtype=torch.float16, device="cuda")
        >>> indices = torch.tensor([0, 0, 1, 0], dtype=torch.int32, device="cuda")
        >>> img = torch.randn(10, 256, dtype=torch.float16, device="cuda")
        >>> aud = torch.randn(5, 256, dtype=torch.float16, device="cuda")
        >>> out = embedding_lookup_multimodal(ids, table, indices, img, aud, 32000, 32001)
        >>> assert out.shape == (4, 256)
    """
    if input_ids.dtype != torch.int32:
        input_ids = input_ids.to(torch.int32)
    if not input_ids.is_contiguous():
        input_ids = input_ids.contiguous()
    if not embedding_table.is_contiguous():
        embedding_table = embedding_table.contiguous()
    if not multimodal_indices.is_contiguous():
        multimodal_indices = multimodal_indices.contiguous()

    # Handle optional image_embeds and audio_embeds [1]
    if image_embeds is None:
        image_embeds = torch.empty(0, embedding_table.shape[1], dtype=embedding_table.dtype, device=input_ids.device)
    else:
        if not image_embeds.is_contiguous():
            image_embeds = image_embeds.contiguous()

    if audio_embeds is None:
        audio_embeds = torch.empty(0, embedding_table.shape[1], dtype=embedding_table.dtype, device=input_ids.device)
    else:
        if not audio_embeds.is_contiguous():
            audio_embeds = audio_embeds.contiguous()

    num_tokens = input_ids.shape[0]
    hidden_size = embedding_table.shape[1]
    output = torch.empty(num_tokens, hidden_size, dtype=embedding_table.dtype, device=input_ids.device)

    _embedding_lookup_multimodal_kernel(
        output, input_ids, embedding_table, multimodal_indices,
        image_embeds, audio_embeds, image_token_id, audio_token_id
    )
    return output