"""ModelRunner runs the forward passes of the models.

pymllm's single-GPU inference architecture.  Handles:

* Model loading (HuggingFace checkpoint via ``transformers``)
* KV-cache memory pool initialisation
* Attention backend setup (FlashInfer)
* Forward pass dispatch (extend / decode / idle)
* Token sampling from logits

Typical lifecycle::

    runner = ModelRunner(server_config, model_config)
    runner.initialize()

    # --- inside the inference loop ---
    forward_batch = runner.prepare_forward_batch_decode(...)
    logits_output = runner.forward(forward_batch)
    next_token_ids = runner.sample(logits_output, forward_batch)

Typical data flow
-----------------
    SchedulerProcess builds a batch dict
        ↓
    ModelRunnerProcess calls ModelRunner.forward(forward_batch)
        ↓
    attn_backend.init_forward_metadata(forward_batch)
        ↓
    model.forward(input_ids, positions, forward_batch)
        ↓
    ModelRunner.sample(logits_output, forward_batch)
        ↓
    next_token_ids returned to scheduler
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
from torch import nn

from pymllm.configs import get_global_config
from pymllm.engine.forward_batch import ForwardBatch, ForwardMode
from pymllm.mem_cache.memory_pool import (
    GDNPool,
    KVPool,
    ReqToTokenPool,
    TokenToKVPoolAllocator,
    make_full_attention_net_mem_pool,
    make_req_to_token_pool,
)

if TYPE_CHECKING:
    from pymllm.configs.model_config import ModelConfig
    from pymllm.configs.server_config import ServerConfig
    from pymllm.executor.cuda_graph_runner import CudaGraphRunner
    from pymllm.layers.attention.attention_backend import AttentionBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: GPU memory query
# ---------------------------------------------------------------------------


def get_available_gpu_memory(device: str = "cuda", gpu_id: int = 0) -> float:
    """Return available GPU memory in GB."""
    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    torch.cuda.set_device(gpu_id)
    free, _ = torch.cuda.mem_get_info(gpu_id)
    return free / (1 << 30)


def get_total_gpu_memory(device: str = "cuda", gpu_id: int = 0) -> float:
    """Return total GPU memory in GB."""
    if device != "cuda" or not torch.cuda.is_available():
        return 0.0
    torch.cuda.set_device(gpu_id)
    _, total = torch.cuda.mem_get_info(gpu_id)
    return total / (1 << 30)


# ---------------------------------------------------------------------------
# LogitsProcessorOutput
# ---------------------------------------------------------------------------


@dataclass
class LogitsProcessorOutput:
    """Container for output logits produced by the model's forward pass.

    Attributes
    ----------
    next_token_logits
        Raw logits for the last token of each sequence in the batch,
        shape ``[batch_size, vocab_size]``.
    hidden_states
        Optional hidden states from the model (e.g. for speculative decoding
        or auxiliary loss computation).
    """

    next_token_logits: torch.Tensor  # [batch_size, vocab_size]
    hidden_states: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# ModelRunner
# ---------------------------------------------------------------------------


class ModelRunner:
    """Runs the forward passes of the models.

    This is the core execution component that owns the model, memory pools,
    and attention backend.  It is used by
    :class:`~pymllm.orchestrator.model_runner_process.ModelRunnerProcess` to
    execute batches dispatched by the scheduler.

    Parameters
    ----------
    server_config
        Server runtime configuration.  Falls back to the global singleton
        when ``None``.
    model_config
        Model configuration (wraps a HuggingFace ``PretrainedConfig``).
        Falls back to the global singleton when ``None``.
    gpu_id
        GPU device index to use.
    """

    def __init__(
        self,
        server_config: Optional["ServerConfig"] = None,
        model_config: Optional["ModelConfig"] = None,
        gpu_id: int = 0,
    ):
        cfg = get_global_config()
        self.server_config = server_config or cfg.server
        self.model_config = model_config or cfg.model

        self.gpu_id = gpu_id
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype: torch.dtype = self._resolve_dtype()

        # Set by initialize()
        self.model: Optional[nn.Module] = None
        self.req_to_token_pool: Optional[ReqToTokenPool] = None
        self.token_to_kv_pool: Optional[KVPool] = None
        self.token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None
        self.gdn_pool: Optional[GDNPool] = None
        self.attn_backend: Optional["AttentionBackend"] = None
        self.graph_runner: Optional["CudaGraphRunner"] = None

        # Memory configuration
        self.max_total_num_tokens: int = 0
        self.max_running_requests: int = 0

        # Model metadata (populated after loading)
        self.num_hidden_layers: int = 0
        self.num_attention_heads: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0
        self.hidden_size: int = 0
        self.vocab_size: int = 0
        self.context_len: int = 0

        # KV cache dtype -- same as model dtype by default; may differ for
        # quantised KV caches in the future.
        self.kv_cache_dtype: torch.dtype = self.dtype

        # Forward pass counter (monotonically increasing).
        self.forward_pass_id: int = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Full initialisation: set device, load model, init memory + backend.

        Call this once before any forward pass.
        """
        tic = time.perf_counter()
        logger.info("ModelRunner initialisation begin.")

        # Set device
        if self.device == "cuda":
            torch.cuda.set_device(self.gpu_id)

        # Set default dtype
        torch.set_default_dtype(self.dtype)

        # Load the model
        self.load_model()

        # Extract model metadata from hf_config
        self._extract_model_metadata()

        # Resolve KV-cache dtype
        self._configure_kv_cache_dtype()

        # Initialise memory pools
        self.init_memory_pool()

        # Initialise attention backend
        self.init_attention_backend()

        # Warm up cuBLAS
        if self.device == "cuda":
            self._init_cublas()

        # Capture CUDA graphs (must be after model + pools + backend)
        self.init_cuda_graphs()

        elapsed = time.perf_counter() - tic
        logger.info(
            "ModelRunner initialisation complete. elapsed=%.2f s, "
            "device=%s, dtype=%s, kv_dtype=%s, max_tokens=%d, max_reqs=%d",
            elapsed,
            self.device,
            self.dtype,
            self.kv_cache_dtype,
            self.max_total_num_tokens,
            self.max_running_requests,
        )

    # ------------------------------------------------------------------
    # Dtype resolution
    # ------------------------------------------------------------------

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve the model dtype from configuration."""
        dtype_str = self.server_config.dtype
        if dtype_str == "auto":
            if torch.cuda.is_available():
                if torch.cuda.get_device_capability()[0] >= 8:
                    return torch.bfloat16
                return torch.float16
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        result = dtype_map.get(dtype_str)
        if result is None:
            raise ValueError(f"Unsupported dtype: {dtype_str!r}")
        return result

    def _configure_kv_cache_dtype(self) -> None:
        """Determine the dtype used for KV-cache storage.

        The global ``QuantizationConfig.kv_cache_dtype`` can override the
        model dtype (e.g. ``fp8_e4m3`` for quantised KV caches).  When set
        to ``"auto"`` the model dtype is used as-is.
        """
        cfg = get_global_config()
        kv_dtype_str = cfg.quantization.kv_cache_dtype

        if kv_dtype_str == "auto":
            self.kv_cache_dtype = self.dtype
            return

        kv_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        resolved = kv_dtype_map.get(kv_dtype_str)
        if resolved is None:
            logger.warning(
                "Unrecognised kv_cache_dtype %r, falling back to model dtype.",
                kv_dtype_str,
            )
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = resolved

        logger.info("KV-cache dtype: %s", self.kv_cache_dtype)

    # ------------------------------------------------------------------
    # Model metadata
    # ------------------------------------------------------------------

    def _extract_model_metadata(self) -> None:
        """Extract key model parameters from the HuggingFace config."""
        hf_config = self.model_config.hf_config
        if hf_config is None:
            raise RuntimeError(
                "HuggingFace config not loaded.  "
                "Make sure model_config.hf_config is set before calling "
                "initialize()."
            )

        # Handle text_config for multimodal models
        text_config = getattr(hf_config, "text_config", hf_config)

        self.num_hidden_layers = getattr(text_config, "num_hidden_layers", 0)
        self.num_attention_heads = getattr(text_config, "num_attention_heads", 0)
        self.num_kv_heads = getattr(
            text_config,
            "num_key_value_heads",
            self.num_attention_heads,
        )
        self.head_dim = getattr(
            text_config,
            "head_dim",
            getattr(text_config, "hidden_size", 0) // max(self.num_attention_heads, 1),
        )
        self.hidden_size = getattr(text_config, "hidden_size", 0)
        self.vocab_size = getattr(text_config, "vocab_size", 0)

        # V-head dim may differ from K-head dim (e.g. MLA)
        self.v_head_dim: int = getattr(text_config, "v_head_dim", self.head_dim)

        # Context length
        self.context_len = self.server_config.context_length or getattr(
            text_config, "max_position_embeddings", 4096
        )

        # Hybrid model metadata (GDN layers)
        self.num_gdn_layers: int = getattr(self.model, "num_gdn_layers", 0)
        self.full_attn_layer_ids: set = getattr(
            self.model, "full_attn_layer_ids", set()
        )

        logger.info(
            "Model metadata: layers=%d, q_heads=%d, kv_heads=%d, "
            "head_dim=%d, v_head_dim=%d, hidden=%d, vocab=%d, ctx_len=%d"
            + (", gdn_layers=%d" if self.num_gdn_layers > 0 else ""),
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_kv_heads,
            self.head_dim,
            self.v_head_dim,
            self.hidden_size,
            self.vocab_size,
            self.context_len,
            *([self.num_gdn_layers] if self.num_gdn_layers > 0 else []),
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the model from a HuggingFace checkpoint.

        First checks the pymllm model registry for a custom implementation
        that uses ``RadixAttention``.  If found, instantiates it with the
        HuggingFace config and loads weights via ``load_weights()``.
        Otherwise falls back to ``AutoModelForCausalLM.from_pretrained``.
        """
        tic = time.perf_counter()
        model_path = self.server_config.model_path

        if model_path is None:
            raise RuntimeError("server_config.model_path is not set.")

        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            "Load model begin.  path=%s, avail mem=%.2f GB",
            model_path,
            before_mem,
        )

        # Look up the architecture in the pymllm model registry
        from pymllm.models import _MODEL_REGISTRY, get_model_class

        hf_config = self.model_config.hf_config
        architectures = []
        if hf_config is not None:
            architectures = getattr(hf_config, "architectures", None) or []

        if not architectures:
            supported = ", ".join(sorted(_MODEL_REGISTRY.keys()))
            raise RuntimeError(
                f"Cannot determine model architecture from config. "
                f"Supported architectures: {supported}"
            )

        architecture = architectures[0]
        model_cls = get_model_class(architecture)
        if model_cls is None:
            supported = ", ".join(sorted(_MODEL_REGISTRY.keys()))
            raise RuntimeError(
                f"Architecture {architecture!r} is not supported by pymllm. "
                f"Supported architectures: {supported}"
            )

        logger.info("Using pymllm model class: %s", model_cls.__name__)
        device_str = f"cuda:{self.gpu_id}" if self.device == "cuda" else self.device
        # Use set_default_dtype so parameters created without explicit dtype
        # get the target dtype, while parameters with explicit dtype=torch.float32
        # (e.g. A_log, dt_bias in GDN layers) stay in float32.
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            with torch.device(device_str):
                self.model = model_cls(hf_config)
        finally:
            torch.set_default_dtype(old_dtype)
        self.model.load_weights(self._iter_weights(model_path))
        self.model.eval()

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        weight_mem = before_mem - after_mem
        logger.info(
            "Load model end.  elapsed=%.2f s, type=%s, "
            "weight_mem=%.2f GB, avail mem=%.2f GB",
            time.perf_counter() - tic,
            type(self.model).__name__,
            weight_mem,
            after_mem,
        )

    @staticmethod
    def _iter_weights(model_path) -> "Generator[Tuple[str, torch.Tensor], None, None]":
        """Yield ``(name, tensor)`` pairs from safetensors or ``.bin`` files.

        Prefers safetensors when available; falls back to PyTorch ``.bin``
        files otherwise.
        """
        import glob as _glob
        from pathlib import Path

        model_path = Path(model_path)

        # Prefer safetensors
        st_files = sorted(_glob.glob(str(model_path / "*.safetensors")))
        if st_files:
            from safetensors.torch import load_file

            for fpath in st_files:
                state_dict = load_file(fpath)
                yield from state_dict.items()
                del state_dict
            return

        # Fallback: PyTorch .bin files
        bin_files = sorted(_glob.glob(str(model_path / "*.bin")))
        for fpath in bin_files:
            state_dict = torch.load(fpath, map_location="cpu", weights_only=True)
            yield from state_dict.items()
            del state_dict

    # ------------------------------------------------------------------
    # Memory pool initialisation
    # ------------------------------------------------------------------

    def init_memory_pool(self) -> None:
        """Initialise KV-cache memory pools and request-to-token mapping.

        1. Profiles available GPU memory to determine the maximum number of
           KV-cache token slots (``max_total_num_tokens``).
        2. Derives ``max_running_requests`` from config or heuristic.
        3. Creates :class:`~pymllm.mem_cache.memory_pool.ReqToTokenPool`,
           :class:`~pymllm.mem_cache.memory_pool.KVPool`, and
           :class:`~pymllm.mem_cache.memory_pool.TokenToKVPoolAllocator`.
        """
        logger.info("Initialising memory pools...")

        # Determine max number of tokens in KV cache
        self.max_total_num_tokens = self._profile_max_num_tokens()

        # Determine max running requests
        max_reqs = self.server_config.max_running_requests
        if max_reqs is None:
            max_reqs = min(
                max(
                    int(self.max_total_num_tokens / self.context_len * 512),
                    2048,
                ),
                4096,
            )
        self.max_running_requests = max_reqs

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory for KV cache.  "
                "Try reducing context_length or using a smaller model."
            )

        # Create ReqToTokenPool
        self.req_to_token_pool = make_req_to_token_pool(
            max_reqs=self.max_running_requests,
            max_context_len=self.context_len + 4,  # small padding
            device=self.device,
        )

        # Create KVPool + TokenToKVPoolAllocator
        # Note: layer_num uses num_hidden_layers even for hybrid models
        # because the KV pool is indexed by global layer_id. GDN layers'
        # KV slots are allocated but unused (they use GDNPool instead).
        self.token_to_kv_pool, self.token_to_kv_pool_allocator = (
            make_full_attention_net_mem_pool(
                size=self.max_total_num_tokens,
                layer_num=self.num_hidden_layers,
                k_head_num=self.num_kv_heads,
                k_head_dim=self.head_dim,
                v_head_num=self.num_kv_heads,
                v_head_dim=self.v_head_dim,
                device=self.device,
                dtype=self.kv_cache_dtype,
            )
        )

        # Create GDNPool if hybrid model with GDN layers
        if self.num_gdn_layers > 0:
            hf_config = self.model_config.hf_config
            text_config = getattr(hf_config, "text_config", hf_config)
            gdn_num_k_heads = getattr(text_config, "linear_num_key_heads", 16)
            gdn_num_v_heads = getattr(text_config, "linear_num_value_heads", 32)
            gdn_head_k_dim = getattr(text_config, "linear_key_head_dim", 128)
            gdn_head_v_dim = getattr(text_config, "linear_value_head_dim", 128)
            gdn_conv_kernel = getattr(text_config, "linear_conv_kernel_dim", 4)
            gdn_conv_dim = (
                gdn_num_k_heads * gdn_head_k_dim * 2 + gdn_num_v_heads * gdn_head_v_dim
            )

            self.gdn_pool = GDNPool(
                max_reqs=self.max_running_requests,
                num_gdn_layers=self.num_gdn_layers,
                num_v_heads=gdn_num_v_heads,
                head_k_dim=gdn_head_k_dim,
                head_v_dim=gdn_head_v_dim,
                conv_dim=gdn_conv_dim,
                conv_kernel_size=gdn_conv_kernel,
                device=self.device,
                dtype=self.dtype,
                max_track_slots=self.max_running_requests,
            )

        logger.info(
            "Memory pool initialised: max_tokens=%d, max_reqs=%d, kv_pool=%.2f GB"
            + (", gdn_pool=%.2f GB" if self.gdn_pool is not None else ""),
            self.max_total_num_tokens,
            self.max_running_requests,
            self.token_to_kv_pool._mem_bytes() / (1 << 30),
            *(
                [self.gdn_pool.mem_bytes() / (1 << 30)]
                if self.gdn_pool is not None
                else []
            ),
        )

    def _profile_max_num_tokens(self) -> int:
        """Profile available memory to determine maximum KV-cache tokens.

        If ``server_config.max_total_tokens`` is explicitly set that value
        is used directly.  Otherwise a memory-fraction-based heuristic
        similar to sglang's ``profile_max_num_token`` is applied.
        """
        # If user explicitly set max_total_tokens, use that.
        if self.server_config.max_total_tokens is not None:
            return self.server_config.max_total_tokens

        if self.device != "cuda":
            # For CPU, use a conservative default.
            return 4096

        available_gb = get_available_gpu_memory(self.device, self.gpu_id)

        # Determine memory fraction for static allocation (KV cache).
        mem_fraction = self.server_config.mem_fraction_static
        if mem_fraction is None:
            mem_fraction = 0.85  # default: use 85% of remaining memory

        # Calculate per-token KV cache size in bytes.
        kv_element_size = torch.tensor([], dtype=self.kv_cache_dtype).element_size()
        cell_size = (
            self.num_kv_heads
            * (self.head_dim + self.v_head_dim)  # K + V
            * self.num_hidden_layers
            * kv_element_size
        )

        if cell_size == 0:
            logger.warning(
                "cell_size is 0 (model metadata may be incomplete); "
                "using default max_total_num_tokens=4096"
            )
            return 4096

        rest_memory_bytes = int(available_gb * mem_fraction * (1 << 30))

        # Reserve memory for GDN pool if hybrid model
        if self.num_gdn_layers > 0:
            hf_config = self.model_config.hf_config
            text_config = getattr(hf_config, "text_config", hf_config)
            gdn_num_k_heads = getattr(text_config, "linear_num_key_heads", 16)
            gdn_num_v_heads = getattr(text_config, "linear_num_value_heads", 32)
            gdn_head_k_dim = getattr(text_config, "linear_key_head_dim", 128)
            gdn_head_v_dim = getattr(text_config, "linear_value_head_dim", 128)
            gdn_conv_kernel = getattr(text_config, "linear_conv_kernel_dim", 4)
            gdn_conv_dim = (
                gdn_num_k_heads * gdn_head_k_dim * 2 + gdn_num_v_heads * gdn_head_v_dim
            )

            # Estimate GDN pool memory for max_running_requests
            # Track slots add max_reqs_est extra slots for prefix cache snapshots
            max_reqs_est = (
                min(
                    max(
                        int(rest_memory_bytes / cell_size / self.context_len * 512),
                        2048,
                    ),
                    4096,
                )
                if self.server_config.max_running_requests is None
                else self.server_config.max_running_requests
            )
            pool_size = max_reqs_est + 1 + max_reqs_est  # +track_slots
            recurrent_bytes = (
                self.num_gdn_layers
                * pool_size
                * gdn_num_v_heads
                * gdn_head_v_dim
                * gdn_head_k_dim
                * 4  # float32
            )
            dtype_size = torch.tensor([], dtype=self.dtype).element_size()
            conv_bytes = (
                self.num_gdn_layers
                * pool_size
                * gdn_conv_dim
                * (gdn_conv_kernel - 1)
                * dtype_size
            )
            gdn_pool_bytes = recurrent_bytes + conv_bytes
            rest_memory_bytes -= gdn_pool_bytes
            logger.info(
                "GDN pool memory reservation: %.2f GB",
                gdn_pool_bytes / (1 << 30),
            )

        max_num_tokens = rest_memory_bytes // cell_size

        logger.info(
            "Memory profiling: avail=%.2f GB, fraction=%.2f, "
            "cell_size=%d bytes, max_tokens=%d",
            available_gb,
            mem_fraction,
            cell_size,
            max_num_tokens,
        )

        return max(max_num_tokens, 1)  # at least 1

    # ------------------------------------------------------------------
    # Attention backend
    # ------------------------------------------------------------------

    def init_attention_backend(self) -> None:
        """Initialise the attention backend.

        Creates a :class:`FlashInferAttnBackend` for standard models, or a
        :class:`HybridAttnBackend` (FlashInfer + GDN) for hybrid models.
        """
        from pymllm.layers.attention.flashinfer_backend import FlashInferAttnBackend

        logger.info("Initialising attention backend...")

        flash_backend = FlashInferAttnBackend(
            num_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            kv_cache_dtype=self.kv_cache_dtype,
            q_dtype=self.dtype,
            max_context_len=self.context_len,
            req_to_token=self.req_to_token_pool.req_to_token,
            device=torch.device(self.device),
            max_req_pool_size=self.req_to_token_pool.size,
        )

        if self.gdn_pool is not None:
            from pymllm.layers.attention.gdn_backend import GDNAttnBackend
            from pymllm.layers.attention.hybrid_backend import HybridAttnBackend

            gdn_backend = GDNAttnBackend(
                gdn_pool=self.gdn_pool,
                device=torch.device(self.device),
            )
            self.attn_backend = HybridAttnBackend(
                full_attn_backend=flash_backend,
                gdn_backend=gdn_backend,
                full_attn_layer_ids=self.full_attn_layer_ids,
            )
        else:
            self.attn_backend = flash_backend

        logger.info(
            "Attention backend: %s",
            type(self.attn_backend).__name__,
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def _init_cublas(self) -> None:
        """Run a small matmul to initialise cuBLAS.

        Without this, the first real matmul may incur a significant
        initialisation overhead.
        """
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        _ = a @ b

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    def init_cuda_graphs(self) -> None:
        """Capture CUDA graphs for decode-step acceleration.

        Skipped when:
        * The device is not CUDA.
        * ``server_config.disable_cuda_graph`` is ``True``.
        * The model is not a generation model.
        """
        self.graph_runner = None

        if self.device != "cuda":
            return
        if self.server_config.disable_cuda_graph:
            logger.info("CUDA graphs disabled by config.")
            return
        if not self.is_generation:
            return

        from pymllm.executor.cuda_graph_runner import CudaGraphRunner

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info("Capturing CUDA graphs... avail mem=%.2f GB", before_mem)

        self.graph_runner = CudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            "CUDA graph capture complete. elapsed=%.2f s, "
            "mem usage=%.2f GB, avail mem=%.2f GB",
            time.perf_counter() - tic,
            before_mem - after_mem,
            after_mem,
        )

    # ------------------------------------------------------------------
    # ForwardBatch construction
    # ------------------------------------------------------------------

    def prepare_forward_batch_extend(
        self,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        return_logprob: bool = False,
        top_logprobs_nums: Optional[List[int]] = None,
    ) -> ForwardBatch:
        """Build a :class:`ForwardBatch` for an extend (prefill) pass.

        Parameters
        ----------
        input_ids
            Token IDs for all new tokens, shape ``[total_new_tokens]``.
        req_pool_indices
            Index of each request in ``ReqToTokenPool``,
            shape ``[batch_size]``.
        seq_lens
            Total (prefix + new) length of each sequence,
            shape ``[batch_size]``.
        extend_seq_lens
            Number of new tokens per sequence, shape ``[batch_size]``.
        extend_prefix_lens
            Cached prefix length per sequence, shape ``[batch_size]``.
        out_cache_loc
            KV-pool slot indices for each new token,
            shape ``[total_new_tokens]``.
        return_logprob
            Whether to return per-token log-probabilities.
        top_logprobs_nums
            Number of top log-probs per sequence.
        """
        batch_size = req_pool_indices.shape[0]
        seq_lens_sum = int(seq_lens.sum().item())
        extend_num_tokens = int(extend_seq_lens.sum().item())

        # Compute positions for each token
        positions = _compute_positions(extend_seq_lens, extend_prefix_lens)

        # Compute extend_start_loc (exclusive cumsum of extend_seq_lens)
        extend_start_loc = torch.zeros(
            batch_size, dtype=torch.int32, device=self.device
        )
        if batch_size > 1:
            extend_start_loc[1:] = torch.cumsum(extend_seq_lens[:-1], dim=0).to(
                torch.int32
            )

        return ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens.cpu(),
            positions=positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens.tolist(),
            extend_seq_lens_cpu=extend_seq_lens.tolist(),
            return_logprob=return_logprob,
            top_logprobs_nums=top_logprobs_nums,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
        )

    def prepare_forward_batch_decode(
        self,
        input_ids: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        return_logprob: bool = False,
        top_logprobs_nums: Optional[List[int]] = None,
        mrope_position_deltas: Optional[torch.Tensor] = None,
    ) -> ForwardBatch:
        """Build a :class:`ForwardBatch` for a decode step.

        Parameters
        ----------
        input_ids
            Token IDs (one per sequence), shape ``[batch_size]``.
        req_pool_indices
            Index of each request in ``ReqToTokenPool``,
            shape ``[batch_size]``.
        seq_lens
            Total sequence length of each request, shape ``[batch_size]``.
        out_cache_loc
            KV-pool slot for each sequence's new token,
            shape ``[batch_size]``.
        return_logprob
            Whether to return per-token log-probabilities.
        top_logprobs_nums
            Number of top log-probs per sequence.
        mrope_position_deltas
            Per-request M-RoPE position deltas, shape ``[batch_size]`` (int64).
            Used by multimodal models (e.g. Qwen3-VL) to offset decode-step
            positions by the spatial extent of prefill images.
        """
        batch_size = req_pool_indices.shape[0]
        seq_lens_sum = int(seq_lens.sum().item())

        # For decode, positions = seq_lens - 1 (the new token position)
        positions = (seq_lens - 1).to(torch.int64)

        return ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens.cpu(),
            positions=positions,
            return_logprob=return_logprob,
            top_logprobs_nums=top_logprobs_nums,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool=self.token_to_kv_pool,
            attn_backend=self.attn_backend,
            mrope_position_deltas=mrope_position_deltas,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Run a forward pass through the model.

        Dispatches to the appropriate method based on the batch's
        :attr:`~pymllm.engine.forward_batch.ForwardMode`.  For decode
        batches, automatically uses CUDA-graph replay when a captured
        graph is available.

        Parameters
        ----------
        forward_batch
            The prepared batch (from ``prepare_forward_batch_*``).

        Returns
        -------
        LogitsProcessorOutput
            Contains ``next_token_logits`` of shape
            ``[batch_size, vocab_size]``.
        """
        self.forward_pass_id += 1

        if forward_batch.forward_mode.is_idle():
            return self._forward_idle(forward_batch)

        # Try CUDA graph replay for decode batches.
        if (
            forward_batch.forward_mode.is_decode()
            and self.graph_runner is not None
            and self.graph_runner.can_run(forward_batch)
        ):
            return self.graph_runner.replay(forward_batch)

        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(forward_batch)
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Run a decode forward pass (one new token per sequence).

        Calls ``attn_backend.init_forward_metadata`` followed by
        ``model.forward``.
        """
        self.attn_backend.init_forward_metadata(forward_batch)
        model_output = self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )
        return self._process_logits(model_output, forward_batch)

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Run an extend (prefill) forward pass.

        Calls ``attn_backend.init_forward_metadata`` followed by
        ``model.forward``.
        """
        self.attn_backend.init_forward_metadata(forward_batch)
        model_output = self.model.forward(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
        )
        return self._process_logits(model_output, forward_batch)

    def _forward_idle(
        self,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Return empty logits for an idle batch (no sequences to process)."""
        return LogitsProcessorOutput(
            next_token_logits=torch.empty(
                (0, self.vocab_size),
                dtype=self.dtype,
                device=self.device,
            ),
        )

    # ------------------------------------------------------------------
    # Logits post-processing
    # ------------------------------------------------------------------

    def _process_logits(
        self,
        model_output: Any,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        """Extract last-token logits from model output.

        Handles:
        * A :class:`LogitsProcessorOutput` returned by custom model
          implementations.
        * A ``CausalLMOutput`` (from HuggingFace ``transformers``) with a
          ``.logits`` attribute.
        * A raw ``torch.Tensor`` of logits.
        """
        if isinstance(model_output, LogitsProcessorOutput):
            return model_output

        # Standard HuggingFace output
        if hasattr(model_output, "logits"):
            logits = model_output.logits
        elif isinstance(model_output, torch.Tensor):
            logits = model_output
        else:
            raise TypeError(
                f"Unexpected model output type: {type(model_output)}.  "
                "Expected torch.Tensor or an object with .logits attribute."
            )

        # --- Decode: logits is [bs, 1, vocab] or [bs, vocab] ---
        if forward_batch.forward_mode.is_decode():
            if logits.dim() == 3:
                next_token_logits = logits[:, -1, :]
            else:
                next_token_logits = logits
        else:
            # --- Extend: pick the last token of each sequence ---
            next_token_logits = self._gather_last_token_logits(logits, forward_batch)

        return LogitsProcessorOutput(next_token_logits=next_token_logits)

    def _gather_last_token_logits(
        self,
        logits: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Gather the logits of the last token in each sequence for extend.

        During extend, the model processes all tokens but we only need the
        logits at the last position of each sequence for next-token sampling.
        """
        if logits.dim() == 3:
            # [batch_size, seq_len, vocab_size] from standard HF model
            return logits[:, -1, :]

        # Flat layout [total_tokens, vocab_size]
        if (
            forward_batch.extend_start_loc is not None
            and forward_batch.extend_seq_lens is not None
        ):
            last_indices = (
                forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
            ).long()
            return logits[last_indices]

        # Fallback: last row
        return logits[-1:, :]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        temperatures: Optional[torch.Tensor] = None,
        top_ps: Optional[torch.Tensor] = None,
        top_ks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample next-token IDs from logits.

        Supports per-request temperature, top-p, and top-k.

        Parameters
        ----------
        logits_output
            The logits from :meth:`forward`.
        forward_batch
            The current forward batch.
        temperatures
            Per-request temperature, shape ``[batch_size]``.
        top_ps
            Per-request top-p, shape ``[batch_size]``.
        top_ks
            Per-request top-k, shape ``[batch_size]``.

        Returns
        -------
        torch.Tensor
            Next-token IDs, shape ``[batch_size]``, dtype ``int32``.
        """
        from pymllm.layers.sampling import (
            sampling_from_probs,
            softmax,
            top_k_top_p_sampling_from_probs,
        )

        logits = logits_output.next_token_logits

        if logits.numel() == 0:
            return torch.empty(0, dtype=torch.int32, device=self.device)

        # Greedy path: temperature=0 (or all zeros) → argmax, no sampling.
        if temperatures is not None:
            all_greedy = bool((temperatures < 1e-6).all())
        else:
            all_greedy = False

        if all_greedy:
            return logits.argmax(dim=-1).to(torch.int32)

        # Stochastic path: apply temperature then sample.
        if temperatures is not None:
            probs = softmax(logits, temperature=temperatures)
        else:
            probs = torch.softmax(logits.float(), dim=-1)

        # Apply top-k / top-p sampling if specified
        has_top_k = top_ks is not None
        has_top_p = top_ps is not None

        if has_top_k or has_top_p:
            k = top_ks if has_top_k else logits.shape[-1]
            p = top_ps if has_top_p else 1.0
            next_token_ids = top_k_top_p_sampling_from_probs(probs, k, p)
        else:
            next_token_ids = sampling_from_probs(probs)

        return next_token_ids

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release model and memory resources."""
        logger.info("ModelRunner shutting down...")

        if self.graph_runner is not None:
            self.graph_runner.shutdown()
            self.graph_runner = None
        if self.model is not None:
            del self.model
            self.model = None
        if self.token_to_kv_pool is not None:
            del self.token_to_kv_pool
            self.token_to_kv_pool = None
        if self.token_to_kv_pool_allocator is not None:
            del self.token_to_kv_pool_allocator
            self.token_to_kv_pool_allocator = None
        if self.gdn_pool is not None:
            del self.gdn_pool
            self.gdn_pool = None
        if self.req_to_token_pool is not None:
            del self.req_to_token_pool
            self.req_to_token_pool = None
        self.attn_backend = None

        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("ModelRunner shutdown complete.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_generation(self) -> bool:
        """True if the model is a generation (causal-LM) model."""
        return True

    @property
    def sliding_window_size(self) -> Optional[int]:
        """Sliding-window attention span, or ``None`` for full context."""
        hf_config = self.model_config.hf_config
        if hf_config is None:
            return None
        text_config = getattr(hf_config, "text_config", hf_config)
        return getattr(text_config, "sliding_window", None)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _compute_positions(
    extend_seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token positions for an extend batch.

    For each sequence, positions are
    ``[prefix_len, prefix_len+1, ..., prefix_len+seq_len-1]``.
    The result is a flat 1-D tensor of shape ``[sum(extend_seq_lens)]``.
    """
    device = extend_seq_lens.device
    batch_size = extend_seq_lens.shape[0]
    total_tokens = int(extend_seq_lens.sum().item())

    if total_tokens == 0:
        return torch.empty(0, dtype=torch.int64, device=device)

    positions = torch.empty(total_tokens, dtype=torch.int64, device=device)
    offset = 0
    for i in range(batch_size):
        seq_len = int(extend_seq_lens[i].item())
        prefix_len = int(extend_prefix_lens[i].item())
        if seq_len > 0:
            positions[offset : offset + seq_len] = torch.arange(
                prefix_len,
                prefix_len + seq_len,
                dtype=torch.int64,
                device=device,
            )
            offset += seq_len

    return positions
