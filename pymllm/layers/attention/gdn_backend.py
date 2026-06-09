"""GDN attention backend -- pooled-state GDN computation for hybrid models.

Performs GDN (Gated Delta Net) linear-attention using externalized state
stored in a :class:`~pymllm.mem_cache.memory_pool.GDNPool`.  Supports
both extend (prefill) and decode paths with FlashInfer kernels.

This backend is not used directly; it is wrapped by
:class:`~pymllm.layers.attention.hybrid_backend.HybridAttnBackend`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from pymllm.engine.forward_batch import ForwardBatch
    from pymllm.layers.attention.radix_linear_attention import RadixLinearAttention
    from pymllm.mem_cache.memory_pool import GDNPool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server config: gdn_decode_backend override
# ---------------------------------------------------------------------------


def _get_gdn_decode_backend_override() -> str:
    """Read ``server.gdn_decode_backend`` from GlobalConfig.

    Returns one of: ``"auto"``, ``"flashinfer"``, ``"mllm_kernel"``, ``"pytorch"``.
    """
    try:
        from pymllm.configs import get_global_config
        return get_global_config().server.gdn_decode_backend
    except Exception:
        return "auto"


def _get_gdn_extend_backend_override() -> str:
    """Read GDN extend backend preference.

    Checks (in order):
    1. ``$PYMLLM_GDN_EXTEND_BACKEND`` env var
    2. ``server.gdn_extend_backend`` config field
    3. Falls back to ``"auto"``

    Returns one of: ``"auto"``, ``"flashinfer"``, ``"mllm_kernel"``,
    ``"cuda_chunkwise"``, ``"torch_chunkwise"``, ``"pytorch"``.
    """
    env = os.environ.get("PYMLLM_GDN_EXTEND_BACKEND", "").strip().lower()
    if env in ("auto", "flashinfer", "mllm_kernel", "cuda_chunkwise",
               "cuda_chunkwise_kernel", "torch_chunkwise", "pytorch"):
        return env
    try:
        from pymllm.configs import get_global_config
        return getattr(get_global_config().server, "gdn_extend_backend", "auto")
    except Exception:
        return "auto"


# ---------------------------------------------------------------------------
# mllm-kernel GDN decode (lazy import, SM80+)
# ---------------------------------------------------------------------------

_mllm_gdn_decode = None


def _get_mllm_gdn_decode():
    """Lazy import for mllm-kernel fused GDN decode CUDA kernel."""
    global _mllm_gdn_decode
    if _mllm_gdn_decode is None:
        try:
            from mllm_kernel.cuda.jit.gdn_decode import gdn_decode

            _mllm_gdn_decode = gdn_decode
            logger.info("GDNAttnBackend: [probe] mllm-kernel GDN decode available (SM80+)")
        except (ImportError, RuntimeError) as e:
            logger.info("GDNAttnBackend: [probe] mllm-kernel GDN decode not available: %s", e)
            _mllm_gdn_decode = False
    return _mllm_gdn_decode if _mllm_gdn_decode is not False else None


# ---------------------------------------------------------------------------
# mllm-kernel GDN extend / prefill (lazy import, SM80+)
# ---------------------------------------------------------------------------

_mllm_gdn_extend = None


def _get_mllm_gdn_extend():
    """Lazy import for mllm-kernel fused GDN extend (prefill) CUDA kernel."""
    global _mllm_gdn_extend
    if _mllm_gdn_extend is None:
        try:
            from mllm_kernel.cuda.jit.gdn_extend import gdn_extend
            _mllm_gdn_extend = gdn_extend
            logger.info("GDNAttnBackend: [probe] mllm-kernel GDN extend available (SM80+)")
        except (ImportError, RuntimeError) as e:
            logger.info("GDNAttnBackend: [probe] mllm-kernel GDN extend not available: %s", e)
            _mllm_gdn_extend = False
    return _mllm_gdn_extend if _mllm_gdn_extend is not False else None


# ---------------------------------------------------------------------------
# mllm-kernel GDN chunkwise extend (lazy import + eager compile, SM80+)
# ---------------------------------------------------------------------------

_mllm_gdn_extend_cw = None


def _get_mllm_gdn_extend_chunkwise():
    """Import and eagerly compile the chunkwise GDN extend CUDA kernel.

    Called from ``GDNAttnBackend.__init__`` when ``cuda_chunkwise`` is
    explicitly configured, so the ~15 s JIT cost is paid at model-load time.
    """
    global _mllm_gdn_extend_cw
    if _mllm_gdn_extend_cw is None:
        try:
            from mllm_kernel.cuda.jit.gdn_extend_chunkwise import gdn_extend_chunkwise
            try:
                from mllm_kernel.cuda.jit.gdn_extend_chunkwise import precompile as _cw_precompile
            except ImportError:
                _cw_precompile = None

            if _cw_precompile is not None:
                logger.info(
                    "GDNAttnBackend: [probe] mllm-kernel GDN chunkwise — "
                    "compiling CUDA kernel (first run may take ~15 s on Jetson) …"
                )
                ok = _cw_precompile()
            else:
                logger.info(
                    "GDNAttnBackend: [probe] mllm-kernel GDN chunkwise — "
                    "precompile not available, kernel will compile on first use"
                )
                ok = True

            if ok:
                _mllm_gdn_extend_cw = gdn_extend_chunkwise
                logger.info("GDNAttnBackend: [probe] mllm-kernel GDN chunkwise extend available (SM80+)")
            else:
                _mllm_gdn_extend_cw = False
                logger.warning(
                    "GDNAttnBackend: [probe] mllm-kernel GDN chunkwise extend "
                    "CUDA compilation failed — falling back to pytorch"
                )
        except Exception as e:
            logger.warning(
                "GDNAttnBackend: [probe] mllm-kernel GDN chunkwise extend "
                "import/compile error (%s: %s) — falling back",
                type(e).__name__, e, exc_info=True,
            )
            _mllm_gdn_extend_cw = False
    return _mllm_gdn_extend_cw if _mllm_gdn_extend_cw is not False else None


# ---------------------------------------------------------------------------
# FlashInfer GDN kernel (lazy import)
# ---------------------------------------------------------------------------

_flashinfer_available: Optional[bool] = None
_fi_chunk_gated_delta_rule = None
_fi_gated_delta_rule_decode = None


def _get_flashinfer_gdn():
    """Lazy import for FlashInfer GDN kernels (prefill + decode)."""
    global _flashinfer_available, _fi_chunk_gated_delta_rule, _fi_gated_delta_rule_decode
    if _flashinfer_available is None:
        try:
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            _flashinfer_available = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability()[0] >= 9
            )
            if not _flashinfer_available:
                logger.info(
                    "GDNAttnBackend: [probe] FlashInfer GDN not available (requires SM90+, "
                    "current SM%d%d)", *torch.cuda.get_device_capability()
                )
                return _flashinfer_available, None, None

            from flashinfer.gdn_prefill import chunk_gated_delta_rule
            _fi_chunk_gated_delta_rule = chunk_gated_delta_rule

            try:
                from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
                _fi_gated_delta_rule_decode = gated_delta_rule_decode_pretranspose
                logger.info("GDNAttnBackend: [probe] FlashInfer GDN available (prefill + decode)")
            except ImportError:
                logger.info(
                    "GDNAttnBackend: [probe] FlashInfer GDN partially available "
                    "(prefill only, decode not found)"
                )
        except (ImportError, RuntimeError) as e:
            logger.info(
                "GDNAttnBackend: [probe] FlashInfer GDN not available: %s", e
            )
            _flashinfer_available = False
    return _flashinfer_available, _fi_chunk_gated_delta_rule, _fi_gated_delta_rule_decode


# ---------------------------------------------------------------------------
# GDN gating computation
# ---------------------------------------------------------------------------


def _gdn_gating(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GDN gating factors.

    Returns
    -------
    g     : log-space decay factor: -exp(A_log) * softplus(a + dt_bias)
    beta  : update gate: sigmoid(b)
    """
    g = -torch.exp(A_log) * F.softplus(a + dt_bias)
    beta = torch.sigmoid(b)
    return g, beta


# ---------------------------------------------------------------------------
# Forward metadata
# ---------------------------------------------------------------------------


@dataclass
class GDNForwardMetadata:
    """Per-batch metadata for GDN backend."""

    cache_indices: torch.Tensor  # [batch_size] = req_pool_indices
    cu_seqlens: Optional[torch.Tensor] = None  # extend only


# ---------------------------------------------------------------------------
# GDNAttnBackend
# ---------------------------------------------------------------------------


class GDNAttnBackend:
    """GDN linear-attention backend using pooled states.

    Handles both extend (prefill) and decode paths for GDN layers.
    Uses FlashInfer kernels when available (SM90+), with PyTorch fallback.

    Parameters
    ----------
    gdn_pool
        Pre-allocated :class:`~pymllm.mem_cache.memory_pool.GDNPool`.
    device
        Target device.
    """

    def __init__(self, gdn_pool: "GDNPool", device: torch.device):
        self.gdn_pool = gdn_pool
        self.device = device
        self.forward_metadata: Optional[GDNForwardMetadata] = None

        # Pre-check FlashInfer availability
        self._use_flashinfer, _, _ = _get_flashinfer_gdn()

        # Eagerly compile the raw chunkwise CUDA kernel only when it is
        # explicitly requested (``cuda_chunkwise_kernel``).  The default
        # ``cuda_chunkwise`` / ``auto`` paths now use the faster fully-batched
        # ``torch_chunkwise`` implementation, which needs no JIT compile.
        if _get_gdn_extend_backend_override() == "cuda_chunkwise_kernel":
            _get_mllm_gdn_extend_chunkwise()

        # One-shot flags to log the selected backend on first actual forward call
        self._decode_backend_logged = False
        self._extend_backend_logged = False

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Prepare GDN metadata from the current forward batch."""
        cache_indices = forward_batch.req_pool_indices.to(torch.int64)

        cu_seqlens = None
        if forward_batch.forward_mode.is_extend():
            # Build cu_seqlens from extend_seq_lens
            if forward_batch.extend_seq_lens is not None:
                seq_lens = forward_batch.extend_seq_lens.to(torch.int64)
                cu_seqlens = torch.zeros(
                    len(seq_lens) + 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.cumsum(seq_lens, dim=0, out=cu_seqlens[1:])

        self.forward_metadata = GDNForwardMetadata(
            cache_indices=cache_indices,
            cu_seqlens=cu_seqlens,
        )

    # ------------------------------------------------------------------
    # CUDA-graph interface
    # ------------------------------------------------------------------

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        """Allocate CUDA-graph state for GDN backend.

        The GDN pool buffers are already pre-allocated at fixed addresses,
        so we only need to allocate the metadata tensor.
        """
        self._cuda_graph_cache_indices = torch.zeros(
            (max_bs,), dtype=torch.int64, device=self.device
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Set up GDN metadata for CUDA-graph capture (decode only)."""
        self._cuda_graph_cache_indices[:bs].copy_(
            req_pool_indices[:bs].to(torch.int64)
        )
        self.forward_metadata = GDNForwardMetadata(
            cache_indices=self._cuda_graph_cache_indices[:bs],
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> None:
        """Update GDN metadata for CUDA-graph replay (decode only)."""
        self._cuda_graph_cache_indices[:bs].copy_(
            req_pool_indices[:bs].to(torch.int64)
        )
        self.forward_metadata = GDNForwardMetadata(
            cache_indices=self._cuda_graph_cache_indices[:bs],
        )

    # ------------------------------------------------------------------
    # Forward: decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        layer: "RadixLinearAttention",
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """GDN decode: one new token per request.

        Steps:
        1. Gather conv_state from pool → [bs, conv_dim, K-1]
        2. Conv1d update: shift + weighted sum for 1 new token
        3. Scatter updated conv_state back to pool
        4. SiLU → split q,k,v
        5. FlashInfer gated_delta_rule_decode (or PyTorch fallback)
        """
        metadata = self.forward_metadata
        cache_indices = metadata.cache_indices
        gdn_idx = layer.gdn_layer_idx
        bs = mixed_qkv.shape[0]

        recurrent_buf, conv_buf = self.gdn_pool.get_layer_state(gdn_idx)
        conv_weight = layer.conv_weight  # [conv_dim, kernel_size]
        K = conv_weight.shape[1]

        # --- Conv1d decode: single-token update ---
        conv_state = conv_buf[cache_indices]  # [bs, conv_dim, K-1]
        x = mixed_qkv.unsqueeze(-1)  # [bs, conv_dim, 1]

        new_conv_state = torch.cat([conv_state[:, :, 1:], x], dim=-1)
        full_window = torch.cat([conv_state, x], dim=-1)  # [bs, conv_dim, K]
        conv_out = (full_window * conv_weight.unsqueeze(0)).sum(dim=-1)

        conv_buf[cache_indices] = new_conv_state

        # --- SiLU activation ---
        conv_out = F.silu(conv_out)

        # --- Split q, k, v ---
        key_dim = layer.num_k_heads * layer.head_k_dim
        value_dim = layer.num_v_heads * layer.head_v_dim
        q, k, v = conv_out.split([key_dim, key_dim, value_dim], dim=-1)
        q = q.view(bs, layer.num_k_heads, layer.head_k_dim)
        k = k.view(bs, layer.num_k_heads, layer.head_k_dim)
        v = v.view(bs, layer.num_v_heads, layer.head_v_dim)

        # --- Recurrent update ---
        # Priority (when "auto"): FlashInfer SM90+ > mllm-kernel SM80+ > PyTorch
        # Can be overridden via --server.gdn_decode_backend
        backend = _get_gdn_decode_backend_override()
        use_fi, _, fi_decode = _get_flashinfer_gdn()
        mllm_gdn = _get_mllm_gdn_decode()

        use_flashinfer = (
            (backend in ("auto", "flashinfer"))
            and use_fi and fi_decode is not None
            and mixed_qkv.is_cuda
        )
        use_mllm = (
            (backend in ("auto", "mllm_kernel"))
            and not (backend == "auto" and use_flashinfer)
            and mllm_gdn is not None
            and mixed_qkv.is_cuda
        )

        if backend == "flashinfer" and not use_flashinfer:
            logger.warning("GDNAttnBackend: gdn_decode_backend='flashinfer' requested but unavailable, falling back")
        if backend == "mllm_kernel" and mllm_gdn is None:
            logger.warning("GDNAttnBackend: gdn_decode_backend='mllm_kernel' requested but unavailable, falling back")

        if not self._decode_backend_logged:
            if use_flashinfer:
                selected = "flashinfer"
            elif use_mllm:
                selected = "mllm_kernel"
            else:
                selected = "pytorch"
            logger.info(
                "GDNAttnBackend: [decode] using backend=%s (config=%s)", selected, backend
            )
            self._decode_backend_logged = True

        if use_flashinfer:
            # FlashInfer decode (SM90+)
            query_fi = q.unsqueeze(1)
            key_fi = k.unsqueeze(1)
            value_fi = v.unsqueeze(1)
            a_fi = a.unsqueeze(1)
            b_fi = b.unsqueeze(1)

            state_batch = recurrent_buf[cache_indices]

            output_fi, new_state = fi_decode(
                q=query_fi, k=key_fi, v=value_fi,
                state=state_batch,
                A_log=layer.A_log.detach(),
                a=a_fi, dt_bias=layer.dt_bias.detach(), b=b_fi,
                scale=None, output=None, use_qk_l2norm=True,
            )

            recurrent_buf[cache_indices] = new_state
            output = output_fi.squeeze(1)

        elif use_mllm:
            # mllm-kernel fused CUDA decode (SM80+)
            output = mllm_gdn(
                q, k, v, a, b,
                layer.A_log, layer.dt_bias,
                recurrent_buf, cache_indices,
            )

        else:
            # PyTorch fallback
            g, beta = _gdn_gating(a, b, layer.A_log, layer.dt_bias)
            output = self._decode_pytorch_fallback(
                q, k, v, g, beta, recurrent_buf, cache_indices, layer
            )

        return output.reshape(bs, value_dim)

    def _decode_pytorch_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_buf: torch.Tensor,
        cache_indices: torch.Tensor,
        layer: "RadixLinearAttention",
    ) -> torch.Tensor:
        """Pure PyTorch decode fallback for GDN with delta rule and L2 norm.

        Matches the sglang Triton kernel (fused_sigmoid_gating_delta_rule_update):
          state *= exp(g)                      # decay
          v_delta = v - state @ k              # delta rule
          v_delta *= beta                      # gating
          state += v_delta outer k             # state update
          output  = state @ q                  # readout
        """
        bs = q.shape[0]
        num_v_heads = layer.num_v_heads
        num_k_heads = layer.num_k_heads

        # GQA: expand k/q heads to match v heads
        if num_k_heads != num_v_heads:
            repeats = num_v_heads // num_k_heads
            q = q.repeat_interleave(repeats, dim=1)
            k = k.repeat_interleave(repeats, dim=1)

        # All computation in float32 (state is float32, avoids dtype mismatch)
        orig_dtype = q.dtype
        q = q.float()
        k = k.float()
        v = v.float()

        # L2 normalize q and k per-head (matching use_qk_l2norm_in_kernel=True)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        q = q * (q.shape[-1] ** -0.5)

        decay = torch.exp(g.float())    # [bs, num_v_heads]
        beta_f = beta.float()           # [bs, num_v_heads]

        outputs = []
        for i in range(bs):
            idx = cache_indices[i]
            state = recurrent_buf[idx]  # [H, V, K] float32

            # Decay
            state = state * decay[i].unsqueeze(-1).unsqueeze(-1)

            k_i = k[i]        # [H, K]
            v_i = v[i]        # [H, V]
            b_i = beta_f[i]   # [H]
            q_i = q[i]        # [H, K]

            # Delta rule: v_delta = v - state @ k
            v_delta = v_i - torch.bmm(state, k_i.unsqueeze(-1)).squeeze(-1)
            v_delta = v_delta * b_i.unsqueeze(-1)  # gating

            # State update: state += v_delta ⊗ k  (outer product in [V, K] layout)
            state = state + v_delta.unsqueeze(-1) * k_i.unsqueeze(-2)
            recurrent_buf[idx] = state

            # Output: o = state @ q
            o_t = torch.bmm(state, q_i.unsqueeze(-1)).squeeze(-1)  # [H, V]
            outputs.append(o_t)

        return torch.stack(outputs, dim=0).to(orig_dtype)  # [bs, H, V]

    # ------------------------------------------------------------------
    # Forward: extend (prefill)
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        layer: "RadixLinearAttention",
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """GDN extend (prefill): multi-token per request.

        Steps:
        1. Gather conv_state from pool for each request
        2. Per-request causal conv1d
        3. Scatter new conv_state back to pool
        4. SiLU → split q,k,v → gating
        5. FlashInfer chunk_gated_delta_rule (or PyTorch fallback)
        6. Scatter final recurrent state back to pool
        """
        metadata = self.forward_metadata
        cache_indices = metadata.cache_indices
        cu_seqlens = metadata.cu_seqlens
        gdn_idx = layer.gdn_layer_idx
        total_tokens = mixed_qkv.shape[0]

        recurrent_buf, conv_buf = self.gdn_pool.get_layer_state(gdn_idx)
        conv_weight = layer.conv_weight  # [conv_dim, kernel_size]
        K = conv_weight.shape[1]
        batch_size = cache_indices.shape[0]

        key_dim = layer.num_k_heads * layer.head_k_dim
        value_dim = layer.num_v_heads * layer.head_v_dim

        # --- Per-request causal conv1d ---
        # Vectorised: unfold + weighted sum (depthwise causal conv) with no
        # Python kernel loop.  Sequence boundaries are read once on the host
        # (avoids a CUDA sync for every request inside the layer hot loop), and
        # the unfold/sum form is faster than grouped F.conv1d for the very wide
        # (conv_dim=6144) depthwise filter used here.
        conv_out = torch.empty_like(mixed_qkv)  # [total_tokens, conv_dim]
        cu_host = cu_seqlens.tolist()

        for i in range(batch_size):
            start = int(cu_host[i])
            end = int(cu_host[i + 1])
            seq_len = end - start
            if seq_len == 0:
                continue

            idx = cache_indices[i]
            x = mixed_qkv[start:end]            # [seq_len, conv_dim]
            prev_state = conv_buf[idx]          # [conv_dim, K-1]

            # [K-1 + seq_len, conv_dim] = prev state ++ new tokens
            x_padded = torch.cat([prev_state.t(), x], dim=0)

            # Save new conv state (last K-1 rows -> [conv_dim, K-1]).
            conv_buf[idx] = x_padded[-(K - 1):].t()

            # Causal conv: out[t] = sum_k x_padded[t+k] * w[:, k].
            # unfold builds a [seq_len, conv_dim, K] sliding window in one shot.
            windows = x_padded.unfold(0, K, 1)              # [seq_len, conv_dim, K]
            conv_out[start:end] = (windows * conv_weight).sum(dim=-1)

        # --- SiLU activation ---
        conv_out = F.silu(conv_out)

        # --- Split q, k, v ---
        q, k, v = conv_out.split([key_dim, key_dim, value_dim], dim=-1)
        q = q.view(total_tokens, layer.num_k_heads, layer.head_k_dim)
        k = k.view(total_tokens, layer.num_k_heads, layer.head_k_dim)
        v = v.view(total_tokens, layer.num_v_heads, layer.head_v_dim)

        # --- Recurrent computation ---
        # Priority (when "auto"):
        #   FlashInfer SM90+ > mllm_kernel SM80+ > cuda_chunkwise SM80+ > PyTorch
        # Override via server.gdn_extend_backend or $PYMLLM_GDN_EXTEND_BACKEND
        backend        = _get_gdn_extend_backend_override()
        use_fi, fi_prefill, _ = _get_flashinfer_gdn()
        mllm_extend    = _get_mllm_gdn_extend()
        mllm_extend_cw = _get_mllm_gdn_extend_chunkwise()

        use_fi_extend = (
            (backend in ("auto", "flashinfer"))
            and use_fi and fi_prefill is not None and mixed_qkv.is_cuda
        )
        use_mllm_extend = (
            not use_fi_extend
            and (backend in ("auto", "mllm_kernel"))
            and mllm_extend is not None and mixed_qkv.is_cuda
        )
        # Fully-batched PyTorch chunkwise (cuBLAS) — the fastest + most
        # numerically faithful prefill path on SM80-89 (no FlashInfer).  It
        # supersedes the sequential ``cuda_chunkwise`` CUDA kernel, so an
        # explicit ``cuda_chunkwise`` request is auto-upgraded to it.  The raw
        # CUDA kernel stays reachable via ``cuda_chunkwise_kernel`` for A/B.
        use_torch_chunkwise = (
            mixed_qkv.is_cuda
            and not use_fi_extend
            and not use_mllm_extend
            and backend in ("auto", "cuda_chunkwise", "torch_chunkwise")
        )
        use_cuda_chunkwise = (
            (not use_torch_chunkwise)
            and backend == "cuda_chunkwise_kernel"
            and mllm_extend_cw is not None
            and mixed_qkv.is_cuda
        )

        if not self._extend_backend_logged:
            selected = (
                "flashinfer"      if use_fi_extend       else
                "mllm_kernel"     if use_mllm_extend     else
                "torch_chunkwise" if use_torch_chunkwise else
                "cuda_chunkwise"  if use_cuda_chunkwise  else
                "pytorch"
            )
            logger.info(
                "GDNAttnBackend: [extend] using backend=%s (config=%s)", selected, backend
            )
            self._extend_backend_logged = True

        if use_fi_extend:
            g, beta = _gdn_gating(a, b, layer.A_log, layer.dt_bias)
            init_state = recurrent_buf[cache_indices].to(torch.float32)
            alpha = torch.exp(g.to(torch.float32))
            beta_f32 = beta.to(torch.float32)
            # FlashInfer's use_qk_l2norm_in_kernel is silently ignored —
            # pre-normalise q/k here to match sglang's approach.
            q_fi = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
            k_fi = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
            q_fi = q_fi * (q_fi.shape[-1] ** -0.5)
            output, final_state = fi_prefill(
                q=q_fi.contiguous(), k=k_fi.contiguous(), v=v.contiguous(),
                g=alpha, beta=beta_f32,
                initial_state=init_state, output_final_state=True,
                cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=False,
            )
            recurrent_buf[cache_indices] = final_state.to(recurrent_buf.dtype)

        elif use_mllm_extend:
            # Sequential recurrent scan — all fused in one CUDA kernel.
            output = mllm_extend(
                q, k, v, a, b,
                layer.A_log, layer.dt_bias,
                recurrent_buf, cache_indices, cu_seqlens,
            )

        elif use_torch_chunkwise:
            # Fully-batched chunkwise (cuBLAS) — parallel across chunks/heads.
            from pymllm.layers.attention.gdn_chunkwise import (
                gdn_extend_chunkwise_torch,
            )
            output = gdn_extend_chunkwise_torch(
                q, k, v, a, b,
                layer.A_log, layer.dt_bias,
                recurrent_buf, cache_indices, cu_seqlens,
            )

        elif use_cuda_chunkwise:
            # Chunkwise WY-parallel CUDA kernel (SM80+).
            output = mllm_extend_cw(
                q, k, v, a, b,
                layer.A_log, layer.dt_bias,
                recurrent_buf, cache_indices, cu_seqlens,
            )

        else:
            g, beta = _gdn_gating(a, b, layer.A_log, layer.dt_bias)
            output = self._extend_pytorch_fallback(
                q, k, v, g, beta, recurrent_buf, cache_indices, cu_seqlens, layer
            )

        return output.reshape(total_tokens, value_dim)

    def _extend_pytorch_fallback(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        recurrent_buf: torch.Tensor,
        cache_indices: torch.Tensor,
        cu_seqlens: torch.Tensor,
        layer: "RadixLinearAttention",
    ) -> torch.Tensor:
        """Pure PyTorch extend fallback for GDN with delta rule and L2 norm."""
        total_tokens = q.shape[0]
        num_v_heads = layer.num_v_heads
        num_k_heads = layer.num_k_heads
        head_v_dim = layer.head_v_dim
        batch_size = cache_indices.shape[0]

        # All computation in float32
        orig_dtype = q.dtype
        q = q.float()
        k = k.float()
        v = v.float()

        # L2 normalize q and k per-head
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)
        q = q * (q.shape[-1] ** -0.5)

        # GQA expansion
        if num_k_heads != num_v_heads:
            repeats = num_v_heads // num_k_heads
            q = q.repeat_interleave(repeats, dim=1)
            k = k.repeat_interleave(repeats, dim=1)

        output = torch.zeros(
            total_tokens, num_v_heads, head_v_dim,
            device=q.device, dtype=torch.float32,
        )

        for i in range(batch_size):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start
            if seq_len == 0:
                continue

            idx = cache_indices[i]
            q_seq = q[start:end]
            k_seq = k[start:end]
            v_seq = v[start:end]
            g_seq = g[start:end]
            beta_seq = beta[start:end]

            decay = torch.exp(g_seq.float())   # [seq_len, H]
            beta_f = beta_seq.float()           # [seq_len, H]
            state = recurrent_buf[idx].clone()  # [H, V, K] float32

            seq_outputs = []
            for t in range(seq_len):
                # Decay
                state = state * decay[t].unsqueeze(-1).unsqueeze(-1)

                k_t = k_seq[t]        # [H, K]
                v_t = v_seq[t]        # [H, V]
                b_t = beta_f[t]       # [H]
                q_t = q_seq[t]        # [H, K]

                # Delta rule: v_delta = v - state @ k
                v_delta = v_t - torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)
                v_delta = v_delta * b_t.unsqueeze(-1)

                # State update
                state = state + v_delta.unsqueeze(-1) * k_t.unsqueeze(-2)

                # Output
                o_t = torch.bmm(state, q_t.unsqueeze(-1)).squeeze(-1)
                seq_outputs.append(o_t)

            recurrent_buf[idx] = state
            output[start:end] = torch.stack(seq_outputs, dim=0)

        return output.to(orig_dtype)

    # ------------------------------------------------------------------
    # Dispatch entry point
    # ------------------------------------------------------------------

    def forward_gdn(
        self,
        layer: "RadixLinearAttention",
        forward_batch: "ForwardBatch",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Route to decode or extend based on forward mode."""
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(layer, forward_batch, mixed_qkv, a, b)
        else:
            return self.forward_extend(layer, forward_batch, mixed_qkv, a, b)
