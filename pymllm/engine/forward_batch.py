"""ForwardMode and ForwardBatch for pymllm.

Simplified forward-batch abstraction: no speculative decoding, no
encoder-decoder support, and no distributed-attention complexity (DP/TP
head splitting is handled at the layer level by the model code, not here).

Typical data flow
-----------------
   ModelRunner builds a ForwardBatch
       ↓
   attn_backend.init_forward_metadata(forward_batch)
       ↓
   model.forward(input_ids, positions, forward_batch)
       ↓
   RadixAttention.forward(q, k, v, forward_batch)
       ↓
   forward_batch.attn_backend.forward(q, k, v, layer, forward_batch)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from pymllm.layers.attention.attention_backend import AttentionBackend
    from pymllm.mem_cache.memory_pool import KVPool, ReqToTokenPool


# ---------------------------------------------------------------------------
# ForwardMode
# ---------------------------------------------------------------------------


class ForwardMode(IntEnum):
    """Describes what kind of forward pass is being performed.

    Covers standard prefill / decode inference without speculative decoding.
    """

    # Prefill / extend: process new tokens.  The KV cache of the prefix (if
    # any) is already populated (e.g. shared system-prompt via radix cache).
    EXTEND = auto()

    # Decode: generate exactly one new token per sequence.
    DECODE = auto()

    # Mixed: a chunked-prefill batch that contains both extend and decode
    # sequences simultaneously.
    MIXED = auto()

    # Idle: no sequences to process (used with data-parallel workers when some
    # ranks have no allocated sequences).
    IDLE = auto()

    # ---- helpers ----

    def is_extend(self) -> bool:
        """True for EXTEND or MIXED (i.e. any prefill-style pass)."""
        return self in (ForwardMode.EXTEND, ForwardMode.MIXED)

    def is_prefill(self) -> bool:
        """Alias for ``is_extend()``."""
        return self.is_extend()

    def is_decode(self) -> bool:
        return self == ForwardMode.DECODE

    def is_mixed(self) -> bool:
        return self == ForwardMode.MIXED

    def is_idle(self) -> bool:
        return self == ForwardMode.IDLE

    def is_decode_or_idle(self) -> bool:
        return self == ForwardMode.DECODE or self == ForwardMode.IDLE


# ---------------------------------------------------------------------------
# ForwardBatch
# ---------------------------------------------------------------------------


@dataclass
class ForwardBatch:
    """All tensors required by a single forward pass through the model.

    Parameters
    ----------
    forward_mode
        The kind of pass being performed (EXTEND / DECODE / MIXED / IDLE).
    batch_size
        Number of sequences in the batch.
    input_ids
        Token ids for every position in the batch, shape ``[num_tokens]``.
        For decode, ``num_tokens == batch_size``; for extend,
        ``num_tokens == extend_num_tokens``.
    req_pool_indices
        Index of each sequence in ``ReqToTokenPool``, shape ``[batch_size]``
        (int32 or int64, on the target device).
    seq_lens
        Total (prefix + new) length of each sequence, shape ``[batch_size]``
        (int32).
    out_cache_loc
        KV-pool slot that each *output* token is written to, shape
        ``[num_tokens]`` (int64).
    seq_lens_sum
        Python ``int`` equal to ``seq_lens.sum()``.  Cached to avoid repeated
        device-to-host syncs.
    seq_lens_cpu
        CPU copy of ``seq_lens`` (optional; used by some attention backends
        for plan computation without a device sync).
    positions
        Token position for each input token, shape ``[num_tokens]``
        (int32 or int64).
    extend_num_tokens
        Total number of new (non-prefix) tokens across the batch.  Only set
        during EXTEND / MIXED passes.
    extend_seq_lens
        Number of *new* tokens for each sequence, shape ``[batch_size]``
        (int32).  Only set during EXTEND / MIXED.
    extend_prefix_lens
        Length of the already-cached prefix for each sequence,
        shape ``[batch_size]`` (int32).  Only set during EXTEND / MIXED.
    extend_start_loc
        Cumulative start offset of each sequence in the flattened extend
        token stream, shape ``[batch_size]`` (int32).
    extend_prefix_lens_cpu
        CPU list mirror of ``extend_prefix_lens``.
    extend_seq_lens_cpu
        CPU list mirror of ``extend_seq_lens``.
    return_logprob
        Whether to compute per-token log-probabilities.
    top_logprobs_nums
        Number of top log-probs to return per sequence (None or list of ints).
    req_to_token_pool
        Reference to the ``ReqToTokenPool`` (set by the model runner).
    token_to_kv_pool
        Reference to the ``KVPool`` (set by the model runner).
    attn_backend
        The attention backend to use (set by the model runner before calling
        ``model.forward``).
    """

    # ---- required fields (positional) ----
    forward_mode: ForwardMode
    batch_size: int
    input_ids: torch.Tensor  # [num_tokens]
    req_pool_indices: torch.Tensor  # [batch_size]   int32/int64
    seq_lens: torch.Tensor  # [batch_size]   int32
    out_cache_loc: torch.Tensor  # [num_tokens]   int64
    seq_lens_sum: int  # python int

    # ---- optional metadata ----

    # CPU mirror of seq_lens
    seq_lens_cpu: Optional[torch.Tensor] = None

    # Position encoding – shape [num_tokens], int32 or int64
    positions: Optional[torch.Tensor] = None

    # ---- extend / prefill specific ----
    extend_num_tokens: Optional[int] = None
    extend_seq_lens: Optional[torch.Tensor] = None  # [batch_size] int32
    extend_prefix_lens: Optional[torch.Tensor] = None  # [batch_size] int32
    extend_start_loc: Optional[torch.Tensor] = None  # [batch_size] int32
    extend_prefix_lens_cpu: Optional[List[int]] = None
    extend_seq_lens_cpu: Optional[List[int]] = None

    # ---- logprob options ----
    return_logprob: bool = False
    top_logprobs_nums: Optional[List[int]] = None

    # ---- memory pools (set by model runner) ----
    req_to_token_pool: Optional["ReqToTokenPool"] = None
    token_to_kv_pool: Optional["KVPool"] = None

    # ---- attention backend (set by model runner) ----
    attn_backend: Optional["AttentionBackend"] = None
