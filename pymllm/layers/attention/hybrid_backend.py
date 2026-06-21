"""Hybrid attention backend -- FlashInfer + GDN for hybrid architectures.

Wraps a :class:`FlashInferAttnBackend` (for full-attention layers) and a
:class:`GDNAttnBackend` (for GDN linear-attention layers).  Dispatches
based on layer type:

* ``RadixAttention`` calls → delegated to ``full_attn_backend``
* ``RadixLinearAttention`` calls (via ``forward_gdn``) → delegated to ``gdn_backend``

CUDA-graph compatible: delegates all graph lifecycle methods to both
sub-backends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Set

import torch

from pymllm.layers.attention.attention_backend import AttentionBackend

if TYPE_CHECKING:
    from pymllm.engine.forward_batch import ForwardBatch, ForwardMode
    from pymllm.layers.attention.flashinfer_backend import FlashInferAttnBackend
    from pymllm.layers.attention.gdn_backend import GDNAttnBackend
    from pymllm.layers.attention.radix_attention import RadixAttention
    from pymllm.layers.attention.radix_linear_attention import RadixLinearAttention

logger = logging.getLogger(__name__)


class HybridAttnBackend(AttentionBackend):
    """Composite attention backend for hybrid full-attention + GDN models.

    Parameters
    ----------
    full_attn_backend
        FlashInfer backend for standard transformer attention layers.
    gdn_backend
        GDN backend for linear-attention layers.
    full_attn_layer_ids
        Set of global layer IDs that use full attention (for logging).
    """

    def __init__(
        self,
        full_attn_backend: "FlashInferAttnBackend",
        gdn_backend: "GDNAttnBackend",
        full_attn_layer_ids: Set[int],
    ):
        self.full_attn_backend = full_attn_backend
        self.gdn_backend = gdn_backend
        self.full_attn_layer_ids = full_attn_layer_ids

        logger.info(
            "HybridAttnBackend created: %d full-attn layers, "
            "%d GDN layers",
            len(full_attn_layer_ids),
            gdn_backend.gdn_pool.num_gdn_layers,
        )

    # ------------------------------------------------------------------
    # Core interface: init_forward_metadata
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Initialize metadata for both sub-backends."""
        self.full_attn_backend.init_forward_metadata(forward_batch)
        self.gdn_backend.init_forward_metadata(forward_batch)

    # ------------------------------------------------------------------
    # Full attention: forward_decode / forward_extend
    # ------------------------------------------------------------------

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
        """Delegate full-attention decode to FlashInfer backend."""
        return self.full_attn_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

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
        """Delegate full-attention extend to FlashInfer backend."""
        return self.full_attn_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache, **kwargs
        )

    # ------------------------------------------------------------------
    # GDN linear attention: forward_gdn
    # ------------------------------------------------------------------

    def forward_gdn(
        self,
        layer: "RadixLinearAttention",
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate GDN computation to the GDN backend."""
        return self.gdn_backend.forward_gdn(
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )

    # ------------------------------------------------------------------
    # CUDA-graph interface: delegate to both sub-backends
    # ------------------------------------------------------------------

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        """Delegate to the full-attention backend."""
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate CUDA-graph state for both sub-backends."""
        self.full_attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)
        self.gdn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: "ForwardMode",
    ) -> None:
        """Set up metadata for CUDA-graph capture in both sub-backends."""
        self.full_attn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            forward_mode=forward_mode,
        )
        self.gdn_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: "ForwardMode",
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        """Update metadata for CUDA-graph replay in both sub-backends."""
        self.full_attn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            forward_mode=forward_mode,
            seq_lens_cpu=seq_lens_cpu,
        )
        self.gdn_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )
