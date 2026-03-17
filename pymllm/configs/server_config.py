from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Centralized runtime configuration for the MLLM server."""

    # --------------------------------------------------------------------- #
    # Model and tokenizer configuration
    # --------------------------------------------------------------------- #
    model_path: Optional[Path] = None
    tokenizer_path: Optional[Path] = None
    tokenizer_mode: Literal["auto", "slow", "fast"] = "auto"
    load_format: Literal["auto", "safetensors"] = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[Path] = None
    context_length: Optional[int] = None
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"

    # --------------------------------------------------------------------- #
    # HTTP / API server
    # --------------------------------------------------------------------- #
    host: str = "127.0.0.1"
    port: int = 30000
    fastapi_root_path: str = ""
    api_key: Optional[str] = None
    admin_api_key: Optional[str] = None
    served_model_name: Optional[str] = None
    file_storage_path: Path = Path("mllm_storage")

    # --------------------------------------------------------------------- #
    # Scheduling and memory
    # --------------------------------------------------------------------- #
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = 1
    max_queued_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    max_prefill_tokens: Optional[int] = None
    schedule_policy: Literal["auto", "fcfs"] = "fcfs"
    schedule_conservativeness: float = 1.0
    sleep_on_idle: bool = False
    stream_interval: int = 1
    stream_output: bool = True

    # --------------------------------------------------------------------- #
    # Device
    # --------------------------------------------------------------------- #
    base_gpu_id: int = 0

    # --------------------------------------------------------------------- #
    # Backend / acceleration
    # --------------------------------------------------------------------- #
    attention_backend: Literal["auto", "flashinfer"] = "auto"
    gdn_decode_backend: Literal["auto", "flashinfer", "mllm_kernel", "pytorch"] = "auto"
    sampling_backend: Optional[str] = None
    disable_cuda_graph: bool = False
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    random_seed: Optional[int] = 42

    # --------------------------------------------------------------------- #
    # Output parsers (reasoning / tool calls)
    # --------------------------------------------------------------------- #
    reasoning_parser: Optional[str] = None   # e.g. "deepseek-r1", "qwen3"
    tool_call_parser: Optional[str] = None   # e.g. "qwen25", "llama3", "hermes"

    # --------------------------------------------------------------------- #
    # Logging and observability
    # --------------------------------------------------------------------- #
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    enable_metrics: bool = False
    show_time_cost: bool = False
    # Log prefill/decode throughput stats every N decode batches (0 = disabled)
    decode_log_interval: int = 40

    # --------------------------------------------------------------------- #
    # Feature switches
    # --------------------------------------------------------------------- #
    enable_shared_queue: bool = False  # Use shared memory queue for fast IPC
    disable_radix_cache: bool = False  # Disable radix-tree prefix caching (uses ChunkCache)
    radix_cache_page_size: int = 1  # Number of tokens per KV-pool page in RadixCache
    enable_mamba_cache: bool = False  # Use MambaRadixCache for SSM state caching

    # CUDA IPC transport for multimodal GPU tensors.
    # Requires enable_shared_queue=True to take effect.
    #
    # Three transport modes (mutually exclusive for GPU tensors):
    #
    #   "default"
    #       GPU tensors are moved to CPU first (GPU→CPU copy), then placed in
    #       POSIX shared memory via share_memory_(). Safe but adds a device copy.
    #
    #   "cuda_ipc"
    #       GPU tensors stay on GPU. Each tensor is wrapped in a
    #       TransportProxyTensor whose __getstate__ calls storage._share_cuda_()
    #       to obtain an IPC handle; the receiver reconstructs via
    #       UntypedStorage._new_shared_cuda(*handle). Simple, but the underlying
    #       GPU allocation is never freed until the sender process exits
    #       (PyTorch limitation) -- can leak GPU memory in long-running services.
    #
    #   "cuda_ipc_pool"  [recommended for production]
    #       GPU tensors are copied into a pre-allocated fixed-size GPU workspace
    #       (MmItemMemoryPool). Each outgoing tensor occupies a "chunk" of the
    #       pool; the chunk's IPC handle is sent via CudaIpcTensorTransportProxy.
    #       After the receiver finishes copying data it increments a shared-memory
    #       sync flag; a background recycler thread in the sender watches these
    #       flags and returns chunks to the available pool. No GPU memory is leaked.
    tensor_transport_mode: str = "default"  # one of: default, cuda_ipc, cuda_ipc_pool

    # Size of the pre-allocated CUDA IPC memory pool in MB.
    # Only used when tensor_transport_mode == "cuda_ipc_pool".
    cuda_ipc_pool_size_mb: int = 512

    # How often (seconds) the pool recycler thread wakes up.
    cuda_ipc_recycle_interval: float = 0.1
    # enable_lora: bool = False
    # max_loaded_loras: Optional[int] = None
    # max_loras_per_batch: int = 8
    # lora_backend: Literal["triton", "csgmv", "torch_native"] = "csgmv"
    # enable_multimodal: bool = False
    # speculative_algorithm: Optional[str] = None
    # speculative_draft_model_path: Optional[Path] = None
    # speculative_num_steps: Optional[int] = None
    # speculative_num_draft_tokens: Optional[int] = None

    # --------------------------------------------------------------------- #
    # Extra
    # --------------------------------------------------------------------- #
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = str(self.model_path)
        self._validate()

    def _validate(self) -> None:
        valid_modes = {"default", "cuda_ipc", "cuda_ipc_pool"}
        if self.tensor_transport_mode not in valid_modes:
            raise ValueError(
                f"`tensor_transport_mode` must be one of {valid_modes}, "
                f"got {self.tensor_transport_mode!r}."
            )
        if self.tensor_transport_mode != "default" and not self.enable_shared_queue:
            raise ValueError(
                "`tensor_transport_mode` != 'default' requires `enable_shared_queue=True`."
            )
        if self.cuda_ipc_pool_size_mb <= 0:
            raise ValueError("`cuda_ipc_pool_size_mb` must be > 0.")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("`port` must be in range [1, 65535].")
        if self.max_prefill_tokens is not None and self.max_prefill_tokens <= 0:
            raise ValueError("`max_prefill_tokens` must be > 0.")
        if self.stream_interval <= 0:
            raise ValueError("`stream_interval` must be > 0.")
        if self.mem_fraction_static is not None and not (
            0.0 < self.mem_fraction_static < 1.0
        ):
            raise ValueError("`mem_fraction_static` must be in (0.0, 1.0).")
        if self.max_running_requests is not None and self.max_running_requests <= 0:
            raise ValueError("`max_running_requests` must be > 0 when set.")
        if self.max_queued_requests is not None and self.max_queued_requests < 0:
            raise ValueError("`max_queued_requests` must be >= 0 when set.")
        if self.radix_cache_page_size < 1:
            raise ValueError("`radix_cache_page_size` must be >= 1.")
        if self.schedule_conservativeness <= 0:
            raise ValueError("`schedule_conservativeness` must be > 0.")
