"""Sampling operations with FlashInfer acceleration and PyTorch fallback.

This module wraps all flashinfer.sampling APIs and provides pure-PyTorch
fallback implementations so that the rest of the codebase can import from
here without worrying about whether FlashInfer is installed.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

try:
    import flashinfer.sampling as _fi_sampling

    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False
    logger.warning("flashinfer not found, falling back to PyTorch sampling kernels")


# ---------------------------------------------------------------------------
# Helper utilities (torch fallback)
# ---------------------------------------------------------------------------


def _resolve_indices(
    data: torch.Tensor, indices: Optional[torch.Tensor]
) -> torch.Tensor:
    """If *indices* is given, gather rows from *data* accordingly."""
    if indices is None:
        return data
    return data[indices.long()]


def _to_scalar_or_tensor(
    value: Union[torch.Tensor, float, int],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Broadcast a scalar or per-batch tensor to shape ``(batch_size,)``."""
    if isinstance(value, (int, float)):
        return torch.full((batch_size,), value, device=device, dtype=torch.float32)
    return value.to(device=device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


def softmax(
    logits: torch.Tensor,
    temperature: Optional[Union[torch.Tensor, float]] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """Safe softmax with optional temperature scaling.

    Parameters
    ----------
    logits : torch.Tensor
        Shape ``(batch_size, num_classes)``.
    temperature : Optional[Union[torch.Tensor, float]]
        Scalar or per-request ``(batch_size,)`` temperature.
    enable_pdl : Optional[bool]
        FlashInfer PDL flag (ignored in fallback).

    Returns
    -------
    torch.Tensor
        Probabilities with the same shape as *logits*.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.softmax(
            logits, temperature=temperature, enable_pdl=enable_pdl
        )

    if temperature is not None:
        if isinstance(temperature, (int, float)):
            logits = logits / temperature
        else:
            logits = logits / temperature.unsqueeze(-1)
    return torch.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# sampling_from_probs
# ---------------------------------------------------------------------------


def sampling_from_probs(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """Category sampling from probabilities.

    Parameters
    ----------
    probs : torch.Tensor
        ``(batch_size, num_classes)`` or ``(unique_batch_size, num_classes)``
        when *indices* is provided.
    indices : Optional[torch.Tensor]
        Maps each output to a row in *probs*.
    deterministic, generator, check_nan, seed, offset
        See FlashInfer docs.

    Returns
    -------
    torch.Tensor
        Sampled token ids, shape ``(batch_size,)``.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.sampling_from_probs(
            probs,
            indices=indices,
            deterministic=deterministic,
            generator=generator,
            check_nan=check_nan,
            seed=seed,
            offset=offset,
        )

    p = _resolve_indices(probs, indices)
    samples = torch.multinomial(p.float(), num_samples=1, generator=generator).squeeze(
        -1
    )
    return samples.to(torch.int32)


# ---------------------------------------------------------------------------
# sampling_from_logits
# ---------------------------------------------------------------------------


def sampling_from_logits(
    logits: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """Category sampling from logits (applies softmax internally).

    Parameters
    ----------
    logits : torch.Tensor
        ``(batch_size, num_classes)``.
    indices, deterministic, generator, check_nan, seed, offset
        See FlashInfer docs.

    Returns
    -------
    torch.Tensor
        Sampled token ids, shape ``(batch_size,)``.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.sampling_from_logits(
            logits,
            indices=indices,
            deterministic=deterministic,
            generator=generator,
            check_nan=check_nan,
            seed=seed,
            offset=offset,
        )

    probs = torch.softmax(logits.float(), dim=-1)
    return sampling_from_probs(
        probs,
        indices=indices,
        deterministic=deterministic,
        generator=generator,
        check_nan=check_nan,
    )


# ---------------------------------------------------------------------------
# top_p_sampling_from_probs
# ---------------------------------------------------------------------------


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """Top-p (nucleus) sampling from probabilities.

    Parameters
    ----------
    probs : torch.Tensor
        ``(batch_size, num_classes)``.
    top_p : Union[torch.Tensor, float]
        Top-p threshold.
    indices, deterministic, generator, check_nan, seed, offset
        See FlashInfer docs.

    Returns
    -------
    torch.Tensor
        Sampled token ids, shape ``(batch_size,)``.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.top_p_sampling_from_probs(
            probs,
            top_p,
            indices=indices,
            deterministic=deterministic,
            generator=generator,
            check_nan=check_nan,
            seed=seed,
            offset=offset,
        )

    p = _resolve_indices(probs, indices).float()
    renormed = _torch_top_p_renorm_probs(p, top_p)
    samples = torch.multinomial(renormed, num_samples=1, generator=generator).squeeze(
        -1
    )
    return samples.to(torch.int32)


# ---------------------------------------------------------------------------
# top_k_sampling_from_probs
# ---------------------------------------------------------------------------


def top_k_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """Top-k sampling from probabilities.

    Parameters
    ----------
    probs : torch.Tensor
        ``(batch_size, num_classes)``.
    top_k : Union[torch.Tensor, int]
        Top-k threshold.
    indices, deterministic, generator, check_nan, seed, offset
        See FlashInfer docs.

    Returns
    -------
    torch.Tensor
        Sampled token ids, shape ``(batch_size,)``.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.top_k_sampling_from_probs(
            probs,
            top_k,
            indices=indices,
            deterministic=deterministic,
            generator=generator,
            check_nan=check_nan,
            seed=seed,
            offset=offset,
        )

    p = _resolve_indices(probs, indices).float()
    renormed = _torch_top_k_renorm_probs(p, top_k)
    samples = torch.multinomial(renormed, num_samples=1, generator=generator).squeeze(
        -1
    )
    return samples.to(torch.int32)


# ---------------------------------------------------------------------------
# min_p_sampling_from_probs
# ---------------------------------------------------------------------------


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """Min-p sampling from probabilities.

    Parameters
    ----------
    probs : torch.Tensor
        ``(batch_size, num_classes)``.
    min_p : Union[torch.Tensor, float]
        Min-p threshold.
    indices, deterministic, generator, check_nan, seed, offset
        See FlashInfer docs.

    Returns
    -------
    torch.Tensor
        Sampled token ids, shape ``(batch_size,)``.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.min_p_sampling_from_probs(
            probs,
            min_p,
            indices=indices,
            deterministic=deterministic,
            generator=generator,
            check_nan=check_nan,
            seed=seed,
            offset=offset,
        )

    p = _resolve_indices(probs, indices).float()
    batch_size = p.shape[0]
    min_p_t = _to_scalar_or_tensor(min_p, batch_size, p.device)
    # min-p: keep tokens whose probability >= min_p * max_prob
    max_probs = p.max(dim=-1, keepdim=True).values  # (B,1)
    threshold = min_p_t.unsqueeze(-1) * max_probs  # (B,1)
    mask = p < threshold
    filtered = p.clone()
    filtered[mask] = 0.0
    # renormalize
    sums = filtered.sum(dim=-1, keepdim=True)
    sums = sums.clamp(min=1e-8)
    filtered = filtered / sums
    samples = torch.multinomial(filtered, num_samples=1, generator=generator).squeeze(
        -1
    )
    return samples.to(torch.int32)


# ---------------------------------------------------------------------------
# top_k_top_p_sampling_from_logits
# ---------------------------------------------------------------------------


def top_k_top_p_sampling_from_logits(
    logits: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """Top-k + top-p sampling from pre-softmax logits.

    Parameters
    ----------
    logits : torch.Tensor
        ``(batch_size, num_classes)``.
    top_k : Union[torch.Tensor, int]
    top_p : Union[torch.Tensor, float]
    filter_apply_order : str
        ``"top_k_first"`` or ``"joint"``.
    indices, deterministic, generator, check_nan, seed, offset
        See FlashInfer docs.

    Returns
    -------
    torch.Tensor
        Sampled token ids, shape ``(batch_size,)``.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.top_k_top_p_sampling_from_logits(
            logits,
            top_k,
            top_p,
            indices=indices,
            filter_apply_order=filter_apply_order,
            deterministic=deterministic,
            generator=generator,
            check_nan=check_nan,
            seed=seed,
            offset=offset,
        )

    probs = torch.softmax(logits.float(), dim=-1)
    return top_k_top_p_sampling_from_probs(
        probs,
        top_k,
        top_p,
        indices=indices,
        filter_apply_order=filter_apply_order,
        deterministic=deterministic,
        generator=generator,
        check_nan=check_nan,
    )


# ---------------------------------------------------------------------------
# top_k_top_p_sampling_from_probs
# ---------------------------------------------------------------------------


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> torch.Tensor:
    """Top-k + top-p sampling from probabilities.

    Parameters
    ----------
    probs : torch.Tensor
        ``(batch_size, num_classes)``.
    top_k : Union[torch.Tensor, int]
    top_p : Union[torch.Tensor, float]
    filter_apply_order : str
        ``"top_k_first"`` or ``"joint"``.
    indices, deterministic, generator, check_nan, seed, offset
        See FlashInfer docs.

    Returns
    -------
    torch.Tensor
        Sampled token ids, shape ``(batch_size,)``.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.top_k_top_p_sampling_from_probs(
            probs,
            top_k,
            top_p,
            indices=indices,
            filter_apply_order=filter_apply_order,
            deterministic=deterministic,
            generator=generator,
            check_nan=check_nan,
            seed=seed,
            offset=offset,
        )

    p = _resolve_indices(probs, indices).float()
    if filter_apply_order == "top_k_first":
        p = _torch_top_k_renorm_probs(p, top_k)
        p = _torch_top_p_renorm_probs(p, top_p)
    else:
        # joint: apply both filters simultaneously
        p = _torch_top_k_renorm_probs(p, top_k)
        p = _torch_top_p_renorm_probs(p, top_p)
    samples = torch.multinomial(p, num_samples=1, generator=generator).squeeze(-1)
    return samples.to(torch.int32)


# ---------------------------------------------------------------------------
# top_p_renorm_probs
# ---------------------------------------------------------------------------


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Renormalize probabilities by top-p thresholding.

    Parameters
    ----------
    probs : torch.Tensor
        ``(batch_size, num_classes)``.
    top_p : Union[torch.Tensor, float]
        Top-p threshold in ``(0, 1)``.

    Returns
    -------
    torch.Tensor
        Renormalized probabilities.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.top_p_renorm_probs(probs, top_p)

    return _torch_top_p_renorm_probs(probs.float(), top_p).to(probs.dtype)


def _torch_top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Pure-torch top-p renormalization (operates on float32)."""
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    if isinstance(top_p, (int, float)):
        mask = cumsum - sorted_probs > top_p
    else:
        top_p_t = top_p.unsqueeze(-1)
        mask = cumsum - sorted_probs > top_p_t

    sorted_probs[mask] = 0.0
    # scatter back
    result = torch.zeros_like(probs)
    result.scatter_(1, sorted_indices, sorted_probs)
    # renormalize
    sums = result.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return result / sums


# ---------------------------------------------------------------------------
# top_k_renorm_probs
# ---------------------------------------------------------------------------


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Renormalize probabilities by top-k thresholding.

    Parameters
    ----------
    probs : torch.Tensor
        ``(batch_size, num_classes)``.
    top_k : Union[torch.Tensor, int]
        Top-k threshold.

    Returns
    -------
    torch.Tensor
        Renormalized probabilities.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.top_k_renorm_probs(probs, top_k)

    return _torch_top_k_renorm_probs(probs.float(), top_k).to(probs.dtype)


def _torch_top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Pure-torch top-k renormalization (operates on float32)."""
    if isinstance(top_k, int):
        # uniform top_k across batch
        topk_vals, _ = torch.topk(probs, top_k, dim=-1)
        threshold = topk_vals[:, -1:]  # (B, 1)
    else:
        # per-request top_k: use sorting
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # gather the k-th value for each row
        k_indices = (top_k.long() - 1).unsqueeze(-1)  # (B, 1)
        threshold = sorted_probs.gather(1, k_indices)  # (B, 1)

    mask = probs < threshold
    filtered = probs.clone()
    filtered[mask] = 0.0
    sums = filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return filtered / sums


# ---------------------------------------------------------------------------
# top_k_mask_logits
# ---------------------------------------------------------------------------


def top_k_mask_logits(
    logits: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Mask logits by top-k thresholding (set non-top-k to -inf).

    Parameters
    ----------
    logits : torch.Tensor
        ``(batch_size, num_classes)``.
    top_k : Union[torch.Tensor, int]
        Top-k threshold.

    Returns
    -------
    torch.Tensor
        Masked logits with the same shape and dtype.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.top_k_mask_logits(logits, top_k)

    if isinstance(top_k, int):
        topk_vals, _ = torch.topk(logits, top_k, dim=-1)
        threshold = topk_vals[:, -1:]
    else:
        sorted_logits, _ = torch.sort(logits, dim=-1, descending=True)
        k_indices = (top_k.long() - 1).unsqueeze(-1)
        threshold = sorted_logits.gather(1, k_indices)

    mask = logits < threshold
    result = logits.clone()
    result[mask] = float("-inf")
    return result


# ---------------------------------------------------------------------------
# chain_speculative_sampling
# ---------------------------------------------------------------------------


def chain_speculative_sampling(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    maybe_output_accepted_token_num: Optional[torch.Tensor] = None,
    maybe_output_emitted_draft_token_num: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None,
    offset: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Speculative sampling for sequence generation.

    Parameters
    ----------
    draft_probs : torch.Tensor
        ``(batch_size, num_speculate_tokens, vocab_size)``.
    draft_token_ids : torch.Tensor
        ``(batch_size, num_speculate_tokens)``.
    target_probs : torch.Tensor
        ``(batch_size, num_speculate_tokens + 1, vocab_size)``.
    maybe_output_accepted_token_num : Optional[torch.Tensor]
        If provided, accepted counts are added in-place.
    maybe_output_emitted_draft_token_num : Optional[torch.Tensor]
        If provided, emitted counts are added in-place.
    deterministic, generator, seed, offset
        See FlashInfer docs.

    Returns
    -------
    output_token_ids : torch.Tensor
        ``(batch_size, num_speculate_tokens + 1)``, rejected slots padded with -1.
    output_accepted_token_num : torch.Tensor
        ``(batch_size,)``.
    output_emitted_draft_token_num : torch.Tensor
        ``(batch_size,)``.
    """
    if _HAS_FLASHINFER:
        return _fi_sampling.chain_speculative_sampling(
            draft_probs,
            draft_token_ids,
            target_probs,
            maybe_output_accepted_token_num=maybe_output_accepted_token_num,
            maybe_output_emitted_draft_token_num=maybe_output_emitted_draft_token_num,
            deterministic=deterministic,
            generator=generator,
            seed=seed,
            offset=offset,
        )

    return _torch_chain_speculative_sampling(
        draft_probs,
        draft_token_ids,
        target_probs,
        maybe_output_accepted_token_num,
        maybe_output_emitted_draft_token_num,
        generator,
    )


def _torch_chain_speculative_sampling(
    draft_probs: torch.Tensor,
    draft_token_ids: torch.Tensor,
    target_probs: torch.Tensor,
    maybe_output_accepted_token_num: Optional[torch.Tensor],
    maybe_output_emitted_draft_token_num: Optional[torch.Tensor],
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-torch chain speculative sampling.

    Implements the rejection-sampling algorithm from
    "Accelerating Large Language Model Decoding with Speculative Sampling"
    (Leviathan et al., 2023).
    """
    batch_size, num_spec, vocab_size = draft_probs.shape
    device = draft_probs.device

    output_ids = torch.full(
        (batch_size, num_spec + 1), -1, dtype=torch.int32, device=device
    )
    accepted_count = torch.zeros(batch_size, dtype=torch.int32, device=device)
    emitted_count = torch.zeros(batch_size, dtype=torch.int32, device=device)

    for b in range(batch_size):
        all_accepted = True
        for t in range(num_spec):
            draft_tok = draft_token_ids[b, t].item()
            p_draft = draft_probs[b, t, draft_tok].item()
            p_target = target_probs[b, t, draft_tok].item()

            # independent acceptance check (for the metric)
            if p_target >= p_draft:
                accepted_count[b] += 1
            else:
                r = torch.rand(1, generator=generator, device=device).item()
                if r < p_target / max(p_draft, 1e-10):
                    accepted_count[b] += 1

            # sequential chain: accept / reject
            if all_accepted:
                r = torch.rand(1, generator=generator, device=device).item()
                if r < min(1.0, p_target / max(p_draft, 1e-10)):
                    output_ids[b, t] = draft_tok
                    emitted_count[b] += 1
                else:
                    # reject: sample from max(0, p_target - p_draft)
                    diff = target_probs[b, t].float() - draft_probs[b, t].float()
                    diff = torch.clamp(diff, min=0.0)
                    dsum = diff.sum()
                    if dsum > 1e-8:
                        diff = diff / dsum
                    else:
                        diff = target_probs[b, t].float()
                        diff = diff / diff.sum().clamp(min=1e-8)
                    resampled = torch.multinomial(
                        diff.unsqueeze(0), num_samples=1, generator=generator
                    ).item()
                    output_ids[b, t] = resampled
                    emitted_count[b] += 1
                    all_accepted = False

        # bonus token (sampled from target at position after last emitted)
        if all_accepted:
            pos = num_spec
            bonus_probs = target_probs[b, pos].float()
            bonus_probs = bonus_probs / bonus_probs.sum().clamp(min=1e-8)
            bonus = torch.multinomial(
                bonus_probs.unsqueeze(0), num_samples=1, generator=generator
            ).item()
            output_ids[b, num_spec] = bonus

    if maybe_output_accepted_token_num is not None:
        maybe_output_accepted_token_num.add_(accepted_count)
    if maybe_output_emitted_draft_token_num is not None:
        maybe_output_emitted_draft_token_num.add_(emitted_count)

    return output_ids, accepted_count, emitted_count


# ---------------------------------------------------------------------------
# Aliases (FlashInfer also exposes these)
# ---------------------------------------------------------------------------
top_p_renorm_prob = top_p_renorm_probs
top_k_renorm_prob = top_k_renorm_probs
