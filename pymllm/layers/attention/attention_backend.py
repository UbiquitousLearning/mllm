"""Abstract base class for pymllm attention backends.

Every concrete backend (FlashInfer, Triton, torch-native, …) must implement
at minimum:

  * ``init_forward_metadata`` – called once per batch before the model forward.
  * ``forward_extend``        – prefill / extend attention.
  * ``forward_decode``        – single-token decode attention.

The public ``forward`` method dispatches to the correct variant based on
``forward_batch.forward_mode``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from pymllm.engine.forward_batch import ForwardBatch, ForwardMode
    from pymllm.layers.attention.radix_attention import RadixAttention


class AttentionBackend(ABC):
    """Abstract base class for attention backends.

    All concrete backends inherit from this class and implement the abstract
    methods below.
    """

    # ------------------------------------------------------------------
    # Core interface – must be implemented by every backend
    # ------------------------------------------------------------------

    @abstractmethod
    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Prepare per-batch metadata before the model's attention layers run.

        For FlashInfer this plans the KV-index arrays and calls
        ``wrapper.begin_forward``; for Triton / torch-native this is a no-op.
        Must be called once per batch *before* ``model.forward``.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_decode(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run attention for a decode step (one new token per sequence)."""
        raise NotImplementedError

    @abstractmethod
    def forward_extend(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Run attention for a prefill / extend step."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Dispatch – shared logic; do not override in normal backends
    # ------------------------------------------------------------------

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Dispatch to ``forward_decode`` or ``forward_extend`` based on mode.

        For IDLE batches a zero-filled output tensor is returned without any
        compute.
        """
        if forward_batch.forward_mode.is_idle():
            # Return empty output without computation.
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )
        else:
            return self.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
            )

    # ------------------------------------------------------------------
    # GDN linear-attention interface (used by HybridAttnBackend)
    # ------------------------------------------------------------------

    def forward_gdn(
        self,
        layer: "RadixLinearAttention",
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Run GDN linear-attention for one layer.

        Only implemented by backends that support hybrid (full + GDN)
        architectures.  The default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support GDN linear attention. "
            "Use HybridAttnBackend for hybrid full+GDN models."
        )

    # ------------------------------------------------------------------
    # Optional CUDA-graph interface
    # ------------------------------------------------------------------

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Fill value used to pad ``seq_lens`` tensors for CUDA-graph capture.

        Most backends use ``1`` (not ``0``) to avoid division-by-zero in
        attention kernels.
        """
        raise NotImplementedError

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate shared CUDA-graph state (buffers reused across captures)."""
        raise NotImplementedError

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: "ForwardMode",
    ) -> None:
        """Set up per-batch metadata for capturing a CUDA graph."""
        raise NotImplementedError

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: "ForwardMode",
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        """Update metadata when replaying a captured CUDA graph."""
        raise NotImplementedError
