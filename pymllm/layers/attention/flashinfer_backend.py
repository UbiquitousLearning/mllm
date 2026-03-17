"""FlashInfer attention backend for pymllm.

  * No model-runner object -- constructor takes explicit scalar / tensor params.
  * No tensor-parallelism head splitting (handled at the model layer level).
  * No speculative decoding support.
  * ``KVPool`` API:
      - ``get_kv_buffer(layer_id)`` returns ``(k_buf, v_buf)`` each shaped
        ``[buf_len, num_heads, head_dim]``.
      - ``set_kv_buffer(layer_id, indices, k, v)`` -- no scale arguments.

Supports:
  * Single-wrapper mode   (full context, no sliding window)
  * Sliding-window mode   (two wrappers: window + full)
  * CUDA-graph capture / replay for decode and target-verify passes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union

import torch

from pymllm.engine.forward_batch import ForwardBatch, ForwardMode
from pymllm.layers.attention.attention_backend import AttentionBackend
from mllm_kernel.cuda.jit.create_kv_indices import create_kv_indices

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FlashInfer import
# ---------------------------------------------------------------------------

_flashinfer_available = False
try:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
    )

    try:
        from flashinfer import fast_decode_plan
        from functools import partial as _partial

        _has_fast_decode_plan = True
    except ImportError:
        _has_fast_decode_plan = False

    from flashinfer.cascade import merge_state

    _flashinfer_available = True
except ImportError:
    logger.warning(
        "flashinfer is not installed; FlashInferAttnBackend will raise "
        "NotImplementedError if used."
    )

# ---------------------------------------------------------------------------
# Global workspace buffer (shared across all FlashInfer wrapper instances)
# ---------------------------------------------------------------------------

_global_workspace_buffer: Optional[torch.Tensor] = None

# Default workspace size (128 MB); can be overridden via environment variable.
_DEFAULT_WORKSPACE_BYTES = int(
    os.environ.get("PYMLLM_FLASHINFER_WORKSPACE_SIZE", 128 * 1024 * 1024)
)

# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------


class WrapperDispatch(Enum):
    """Indicates which wrapper to use for a given attention layer."""

    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class DecodeMetadata:
    """Per-batch metadata for a decode step."""

    decode_wrappers: "List[BatchDecodeWithPagedKVCacheWrapper]"


@dataclass
class PrefillMetadata:
    """Per-batch metadata for a prefill / extend step."""

    prefill_wrappers: "List[BatchPrefillWithPagedKVCacheWrapper]"
    use_ragged: bool
    extend_no_prefix: bool


# ---------------------------------------------------------------------------
# CUDA kernel – build the flat kv_indices array for FlashInfer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper – choose whether to use tensor cores for decode
# ---------------------------------------------------------------------------


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """Return whether FlashInfer decode should use tensor cores.

    For FP8 we always use tensor cores.  For fp16 / bf16 we use them when
    the GQA group size (num_attention_heads / num_kv_heads) is ≥ 4, which
    fuses the head group with the token dimension in the MMA instruction.
    """
    env_override = os.environ.get("PYMLLM_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        return not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads, num_kv_heads
        )
    except (ImportError, AttributeError):
        pass

    gqa_group_size = num_attention_heads // num_kv_heads
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    if kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    return False


# ---------------------------------------------------------------------------
# FlashInferAttnBackend
# ---------------------------------------------------------------------------


class FlashInferAttnBackend(AttentionBackend):
    """FlashInfer-based attention backend for pymllm.

    This class does not depend on a ``ModelRunner`` object.  Instead it takes
    all required configuration explicitly so that it can be constructed
    independently of any particular model runner.

    Parameters
    ----------
    num_heads
        Number of query heads per device (after any TP sharding).
    num_kv_heads
        Number of KV heads per device.
    head_dim
        Per-head dimension for Q and K.
    kv_cache_dtype
        ``torch.dtype`` of the KV cache (e.g. ``torch.float16``).
    q_dtype
        ``torch.dtype`` of the query tensor.
    max_context_len
        Maximum sequence length the model supports.
    req_to_token
        The ``[max_reqs, max_context_len]`` int32 tensor from
        ``ReqToTokenPool.req_to_token``.
    device
        Target device (e.g. ``torch.device("cuda")``)
    max_req_pool_size
        Maximum number of concurrent requests (= ``ReqToTokenPool.size``).
        Used to pre-allocate ``kv_indptr`` / ``kv_last_page_len`` buffers.
    sliding_window_size
        When not ``None``, enables sliding-window attention mode which
        allocates two wrapper sets (window + full context).
    skip_prefill
        When ``True``, skip creating prefill wrappers (for backends that only
        perform decode, e.g. multi-step draft backends).
    kv_indptr_buf
        Optional pre-allocated ``kv_indptr`` buffer.  Used when sharing
        buffers across multiple backend instances (e.g. multi-step draft).
    kv_last_page_len_buf
        Optional pre-allocated ``kv_last_page_len`` buffer.
    init_new_workspace
        When ``True`` allocate a fresh workspace buffer instead of reusing the
        global one.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: torch.dtype,
        q_dtype: torch.dtype,
        max_context_len: int,
        req_to_token: torch.Tensor,
        device: torch.device,
        max_req_pool_size: int,
        sliding_window_size: Optional[int] = None,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        if not _flashinfer_available:
            raise RuntimeError(
                "flashinfer is required for FlashInferAttnBackend but is not "
                "installed.  Run: pip install flashinfer-python"
            )

        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_cache_dtype = kv_cache_dtype
        self.q_dtype = q_dtype
        self.max_context_len = max_context_len
        self.req_to_token = req_to_token
        self.device = device
        self.skip_prefill = skip_prefill

        # Tensor-core preference for decode
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype, num_heads, num_kv_heads
        )

        # Sliding-window / cross-attention wrapper dispatch
        if sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason: Optional[WrapperDispatch] = (
                WrapperDispatch.SLIDING_WINDOW
            )
            self.sliding_window_size: Optional[int] = sliding_window_size
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None
            self.sliding_window_size = None

        # ------------------------------------------------------------------
        # Workspace buffer
        # ------------------------------------------------------------------
        global _global_workspace_buffer
        if _global_workspace_buffer is None:
            _global_workspace_buffer = torch.empty(
                _DEFAULT_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=device,
            )
        if init_new_workspace:
            self.workspace_buffer = torch.empty(
                _DEFAULT_WORKSPACE_BYTES,
                dtype=torch.uint8,
                device=device,
            )
        else:
            self.workspace_buffer = _global_workspace_buffer

        # ------------------------------------------------------------------
        # kv_indptr  [num_wrappers × (max_req_pool_size + 1)]
        # kv_last_page_len  [max_req_pool_size]
        # ------------------------------------------------------------------
        if kv_indptr_buf is None:
            self.kv_indptr: List[torch.Tensor] = [
                torch.zeros((max_req_pool_size + 1,), dtype=torch.int32, device=device)
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_req_pool_size,), dtype=torch.int32, device=device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        # qo_indptr – only needed for prefill
        if not skip_prefill:
            self.qo_indptr: List[torch.Tensor] = [
                torch.zeros((max_req_pool_size + 1,), dtype=torch.int32, device=device)
                for _ in range(self.num_wrappers)
            ]

        # ------------------------------------------------------------------
        # Create FlashInfer wrappers
        # ------------------------------------------------------------------
        self.prefill_wrapper_ragged: Optional[
            "BatchPrefillWithRaggedKVCacheWrapper"
        ] = None
        self.prefill_wrappers_paged: List["BatchPrefillWithPagedKVCacheWrapper"] = []
        self.decode_wrappers: List["BatchDecodeWithPagedKVCacheWrapper"] = []

        if not skip_prefill:
            self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )

        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
                )
            self.decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_tensor_cores=self.decode_use_tensor_cores,
                )
            )

        # ------------------------------------------------------------------
        # Indices updaters
        # ------------------------------------------------------------------
        if not skip_prefill:
            self.indices_updater_prefill = _FlashInferIndicesUpdaterPrefill(self)
        self.indices_updater_decode = _FlashInferIndicesUpdaterDecode(self)

        # Per-batch metadata set by init_forward_metadata
        self.forward_metadata: Optional[Union[DecodeMetadata, PrefillMetadata]] = None

        # CUDA-graph metadata stores
        self.decode_cuda_graph_metadata: dict = {}
        self.prefill_cuda_graph_metadata: dict = {}

    # ------------------------------------------------------------------
    # init_forward_metadata
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        """Prepare FlashInfer wrappers for the current batch.

        Must be called once per batch before the model's ``forward`` method.
        """
        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
            )
            self.forward_metadata = DecodeMetadata(self.decode_wrappers)
        else:
            # Extend / prefill
            prefix_lens = forward_batch.extend_prefix_lens
            extend_no_prefix = (
                forward_batch.extend_prefix_lens_cpu is None
                or not any(forward_batch.extend_prefix_lens_cpu)
            )
            # use_ragged=True: match sglang's default.
            # - extend_no_prefix=True  → ragged-only (pure prefill, no cache)
            # - extend_no_prefix=False → ragged+paged merge (cache hit)
            # The paged wrapper covers only the cached prefix (prefix_lens),
            # the ragged wrapper covers the new extend tokens.  No overlap.
            # NOTE: to avoid a FlashInfer edge-case with 1-token ragged
            # extends, _allocate_extend guarantees extend_len >= 2.
            use_ragged = True

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                extend_no_prefix=extend_no_prefix,
            )

    # ------------------------------------------------------------------
    # forward_extend
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: "RadixAttention",  # noqa: F821
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        from pymllm.layers.attention.radix_attention import RadixAttention

        assert isinstance(layer, RadixAttention)
        meta: PrefillMetadata = self.forward_metadata

        prefill_wrapper_paged = meta.prefill_wrappers[self._get_wrapper_idx(layer)]
        cache_loc = forward_batch.out_cache_loc

        # Write K/V into the pool
        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer.layer_id, cache_loc, k, v
                )

        q_3d = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)

        if not meta.use_ragged:
            # Paged-only path: uses the full KV cache (prefix + extend).
            k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            # Reshape to [buf_len, page_size=1, num_heads, head_dim] for FlashInfer.
            paged_kv = (k_cache.unsqueeze(1), v_cache.unsqueeze(1))

            o = prefill_wrapper_paged.forward(
                q_3d,
                paged_kv,
                causal=not layer.is_cross_attention,
                sm_scale=layer.scaling,
                window_left=layer.sliding_window_size,
                logits_soft_cap=layer.logit_cap if layer.logit_cap > 0 else None,
            )
        else:
            # Ragged path: query attends only to the new (ragged) K/V;
            # prefix K/V is in the paged pool.
            if k is None:
                # Fallback: load K/V from the pool.
                k_buf, v_buf = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )
                k = k_buf
                v = v_buf

            k_3d = k.view(-1, layer.tp_k_head_num, layer.head_dim)
            v_3d = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

            if meta.extend_no_prefix:
                # Pure prefill – no prefix at all.
                o = self.prefill_wrapper_ragged.forward(
                    q_3d,
                    k_3d,
                    v_3d,
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=(layer.logit_cap if layer.logit_cap > 0 else None),
                )
            else:
                # Extend with prefix: merge ragged (new) and paged (prefix).
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q_3d,
                    k_3d,
                    v_3d,
                    causal=True,
                    sm_scale=layer.scaling,
                    logits_soft_cap=(layer.logit_cap if layer.logit_cap > 0 else None),
                )

                k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                    layer.layer_id
                )
                paged_kv = (k_cache.unsqueeze(1), v_cache.unsqueeze(1))
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q_3d,
                    paged_kv,
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=(layer.logit_cap if layer.logit_cap > 0 else None),
                )

                o, _ = merge_state(o1, s1, o2, s2)

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # forward_decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        layer: "RadixAttention",  # noqa: F821
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        from pymllm.layers.attention.radix_attention import RadixAttention

        assert isinstance(layer, RadixAttention)
        meta: DecodeMetadata = self.forward_metadata

        decode_wrapper = meta.decode_wrappers[self._get_wrapper_idx(layer)]
        cache_loc = forward_batch.out_cache_loc

        if k is not None:
            assert v is not None
            if save_kv_cache:
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer.layer_id, cache_loc, k, v
                )

        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        paged_kv = (k_cache.unsqueeze(1), v_cache.unsqueeze(1))

        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            paged_kv,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap if layer.logit_cap > 0 else None,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # CUDA-graph support
    # ------------------------------------------------------------------

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ) -> None:
        """Allocate CUDA-graph shared state buffers."""
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.uint8,
                device=self.device,
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> None:
        """Set up metadata for CUDA-graph capture of a decode step."""
        if not forward_mode.is_decode_or_idle():
            raise ValueError(
                "CUDA-graph capture is only supported for decode / idle modes."
            )

        decode_wrappers = []
        for i in range(self.num_wrappers):
            decode_wrappers.append(
                BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer,
                    "NHD",
                    use_cuda_graph=True,
                    use_tensor_cores=self.decode_use_tensor_cores,
                    paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                    paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                    paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
                )
            )

        seq_lens_sum = seq_lens.sum().item()
        self.indices_updater_decode.update(
            req_pool_indices,
            seq_lens,
            seq_lens.cpu(),
            seq_lens_sum,
            decode_wrappers=decode_wrappers,
        )
        self.decode_cuda_graph_metadata[bs] = decode_wrappers
        self.forward_metadata = DecodeMetadata(decode_wrappers)

        if _has_fast_decode_plan:
            for i in range(self.num_wrappers):
                decode_wrappers[i].begin_forward = _partial(
                    fast_decode_plan, decode_wrappers[i]
                )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        forward_mode: ForwardMode,
        seq_lens_cpu: Optional[torch.Tensor],
    ) -> None:
        """Update metadata when replaying a CUDA graph for decode."""
        if not forward_mode.is_decode_or_idle():
            raise ValueError(
                "CUDA-graph replay is only supported for decode / idle modes."
            )

        self.indices_updater_decode.update(
            req_pool_indices[:bs],
            seq_lens[:bs],
            seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
            seq_lens_sum,
            decode_wrappers=self.decode_cuda_graph_metadata[bs],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_wrapper_idx(self, layer) -> int:
        """Return the wrapper index for the given attention layer."""
        if self.num_wrappers == 1:
            return 0
        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            # Wrapper 0 → sliding window attention.
            # Wrapper 1 → full-context attention.
            return int(layer.sliding_window_size == -1)
        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


# ---------------------------------------------------------------------------
# _FlashInferIndicesUpdaterDecode
# ---------------------------------------------------------------------------


class _FlashInferIndicesUpdaterDecode:
    """Populates ``kv_indptr`` / ``kv_indices`` and calls
    ``wrapper.begin_forward`` before every decode step.
    """

    def __init__(self, backend: FlashInferAttnBackend):
        self.num_qo_heads = backend.num_heads
        self.num_kv_heads = backend.num_kv_heads
        self.head_dim = backend.head_dim
        self.data_type = backend.kv_cache_dtype
        self.q_data_type = backend.q_dtype
        self.sliding_window_size = backend.sliding_window_size
        self.backend = backend

        self.kv_indptr = backend.kv_indptr
        self.kv_last_page_len = backend.kv_last_page_len
        self.req_to_token = backend.req_to_token

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: "List[BatchDecodeWithPagedKVCacheWrapper]",
        kv_start_idx: Optional[torch.Tensor] = None,
    ) -> None:
        if self.backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self._update_sliding_window(
                req_pool_indices,
                seq_lens,
                seq_lens_cpu,
                seq_lens_sum,
                decode_wrappers,
            )
        else:
            # Single-wrapper: full-context decode. Build kv_indptr/kv_indices
            # and call FlashInfer's plan function via the CUDA kernel.
            bs = len(req_pool_indices)
            kv_indptr = self.kv_indptr[0]

            # Fill kv_indptr: prefix sums of paged_kernel_lens.
            kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens, dim=0)
            kv_indptr_sliced = kv_indptr[: bs + 1]

            if seq_lens_cpu is not None:
                seq_lens_sum = int(seq_lens_cpu.sum().item())
            else:
                seq_lens_sum = int(seq_lens.sum().item())

            # Allocate KV indices buffer.
            if decode_wrappers and decode_wrappers[0].is_cuda_graph_enabled:
                kv_indices = decode_wrappers[0]._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    seq_lens_sum, dtype=torch.int32, device=self.req_to_token.device
                )

            # Use high-performance CUDA kernel to populate kv_indices.
            create_kv_indices(
                self.req_to_token,
                req_pool_indices.to(torch.int32),
                seq_lens.to(torch.int32),
                kv_indptr_sliced,
                None,
                kv_indices,
            )

            decode_wrappers = decode_wrappers or self.decode_wrappers
            decode_wrappers[0].begin_forward(
                kv_indptr_sliced,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
            )

    def _update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: "List[BatchDecodeWithPagedKVCacheWrapper]",
    ) -> None:
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding-window attention: clamp to window size + 1
                paged_kernel_lens = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                paged_kernel_lens_sum = int(paged_kernel_lens.sum().item())
                kv_start_idx = seq_lens - paged_kernel_lens
                seq_lens_cpu_tmp = (
                    torch.clamp(seq_lens_cpu, max=self.sliding_window_size + 1)
                    if seq_lens_cpu is not None
                    else None
                )
            else:
                # Full-context attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum
                kv_start_idx = None
                seq_lens_cpu_tmp = seq_lens_cpu

            bs = len(req_pool_indices)
            kv_indptr = self.kv_indptr[wrapper_id]
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr_sliced = kv_indptr[: bs + 1]

            if decode_wrappers and decode_wrappers[wrapper_id].is_cuda_graph_enabled:
                kv_indices = decode_wrappers[wrapper_id]._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum,
                    dtype=torch.int32,
                    device=self.req_to_token.device,
                )

            # High-performance CUDA kernel populates kv_indices from req_to_token.
            create_kv_indices(
                self.req_to_token,
                req_pool_indices.to(torch.int32),
                paged_kernel_lens.to(torch.int32),
                kv_indptr_sliced,
                kv_start_idx.to(torch.int32) if kv_start_idx is not None else None,
                kv_indices,
            )

            decode_wrappers[wrapper_id].begin_forward(
                kv_indptr_sliced,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
            )


# ---------------------------------------------------------------------------
# _FlashInferIndicesUpdaterPrefill
# ---------------------------------------------------------------------------


class _FlashInferIndicesUpdaterPrefill:
    """Populates indices and calls ``wrapper.begin_forward`` before extend."""

    def __init__(self, backend: FlashInferAttnBackend):
        self.num_qo_heads = backend.num_heads
        self.num_kv_heads = backend.num_kv_heads
        self.head_dim = backend.head_dim
        self.data_type = backend.kv_cache_dtype
        self.q_data_type = backend.q_dtype
        self.sliding_window_size = backend.sliding_window_size
        self.backend = backend

        self.kv_indptr = backend.kv_indptr
        self.kv_last_page_len = backend.kv_last_page_len
        self.qo_indptr = backend.qo_indptr
        self.req_to_token = backend.req_to_token
        self.prefill_wrapper_ragged = backend.prefill_wrapper_ragged

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: "List[BatchPrefillWithPagedKVCacheWrapper]",
        use_ragged: bool,
    ) -> None:
        if self.backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self._update_sliding_window(
                req_pool_indices,
                seq_lens,
                seq_lens_cpu,
                seq_lens_sum,
                prefix_lens,
                prefill_wrappers,
                use_ragged,
            )
        else:
            if use_ragged:
                # Merge path: paged covers ONLY the cached prefix so there
                # is no overlap with the ragged (extend) tokens.
                paged_kernel_lens = prefix_lens
                paged_kernel_lens_sum = int(paged_kernel_lens.sum().item())
            else:
                # Paged-only path: covers the full sequence.
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum

            self._call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[0],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx=None,
                kv_indptr=self.kv_indptr[0],
                qo_indptr=self.qo_indptr[0],
                use_ragged=use_ragged,
            )

    def _update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: "List[BatchPrefillWithPagedKVCacheWrapper]",
        use_ragged: bool,
    ) -> None:
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding-window portion uses a limited context window.
                extend_lens = seq_lens - prefix_lens
                paged_kernel_lens = torch.minimum(
                    seq_lens,
                    torch.tensor(self.sliding_window_size, device=seq_lens.device)
                    + extend_lens,
                )
                paged_kernel_lens_sum = int(paged_kernel_lens.sum().item())
                kv_start_idx = seq_lens - paged_kernel_lens
            else:
                # Full-context SWA wrapper: same split as non-SWA.
                if use_ragged:
                    paged_kernel_lens = prefix_lens
                    paged_kernel_lens_sum = int(paged_kernel_lens.sum().item())
                else:
                    paged_kernel_lens = seq_lens
                    paged_kernel_lens_sum = seq_lens_sum
                kv_start_idx = None

            kv_indptr = self.kv_indptr[wrapper_id]
            qo_indptr = self.qo_indptr[wrapper_id]

            self._call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx=kv_start_idx,
                kv_indptr=kv_indptr,
                qo_indptr=qo_indptr,
                use_ragged=use_ragged,
            )

    def _call_begin_forward(
        self,
        wrapper_ragged: "BatchPrefillWithRaggedKVCacheWrapper",
        wrapper_paged: "BatchPrefillWithPagedKVCacheWrapper",
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: Optional[torch.Tensor],
        kv_start_idx: Optional[torch.Tensor],
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
    ) -> None:
        bs = len(seq_lens)

        # Build kv_indptr and kv_indices using the CUDA kernel.
        kv_indptr_sliced = kv_indptr[: bs + 1]
        kv_indptr_sliced[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + 256,
            dtype=torch.int32,
            device=req_pool_indices.device,
        )

        create_kv_indices(
            self.req_to_token,
            req_pool_indices.to(torch.int32),
            paged_kernel_lens.to(torch.int32),
            kv_indptr_sliced,
            kv_start_idx.to(torch.int32) if kv_start_idx is not None else None,
            kv_indices,
        )

        # Build qo_indptr (number of new tokens per sequence).
        if prefix_lens is not None:
            extend_lens = seq_lens - prefix_lens
        else:
            extend_lens = seq_lens
        qo_indptr_sliced = qo_indptr[: bs + 1]
        qo_indptr_sliced[1:] = torch.cumsum(extend_lens, dim=0)

        # Plan the ragged wrapper (new tokens only).
        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr_sliced,
                qo_indptr_sliced,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )

        # Plan the paged wrapper (cached prefix tokens).
        wrapper_paged.begin_forward(
            qo_indptr_sliced,
            kv_indptr_sliced,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            non_blocking=True,
        )
