from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional
from dataclasses import asdict, dataclass, field


@dataclass
class ServerConfig:
    """
    Centralized runtime configuration for the MLLM server.

    The fields are grouped by operational concern so that:
    - CLI args can map directly to this dataclass.
    - YAML/JSON config files can be loaded and validated in one place.
    - future extensions can follow a predictable structure.
    """

    # -------------------------------------------------------------------------
    # Model and tokenizer settings
    # -------------------------------------------------------------------------
    # Required path to the model checkpoint directory or model identifier.
    model_path: Path
    # Optional tokenizer path; when omitted we fall back to `model_path`.
    tokenizer_path: Optional[Path] = None
    # Tokenizer bootstrap strategy:
    # - "auto": infer tokenizer mode from model type.
    # - "slow"/"fast": force a specific tokenizer implementation.
    tokenizer_mode: Literal["auto", "slow", "fast"] = "auto"
    # Number of worker threads/processes used by tokenizer service.
    tokenizer_worker_num: int = 1
    # Skip tokenizer initialization at startup to reduce cold-start latency.
    skip_tokenizer_init: bool = False
    # Model loading format hint for loader backends.
    load_format: Literal["auto", "pt", "safetensors", "gguf"] = "auto"
    # Allow loading custom model code from remote repositories.
    trust_remote_code: bool = False
    # Explicit context length; `None` means infer from model config.
    context_length: Optional[int] = None
    # Model precision policy for weights and activations.
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    # Quantization algorithm to apply at load time.
    quantization: Optional[str] = None
    # KV cache dtype; can differ from model dtype for better memory trade-offs.
    kv_cache_dtype: Literal["auto", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2"] = (
        "auto"
    )
    # HuggingFace revision/commit/tag for deterministic model resolution.
    revision: Optional[str] = None
    # Optional custom directory used to cache downloaded model artifacts.
    download_dir: Optional[Path] = None

    # -------------------------------------------------------------------------
    # HTTP / API server settings
    # -------------------------------------------------------------------------
    # Host address the HTTP server binds to.
    host: str = "127.0.0.1"
    # TCP port exposed by the HTTP server.
    port: int = 30000
    # Optional FastAPI root path when running behind a reverse proxy.
    fastapi_root_path: str = ""
    # API key required by client-facing endpoints.
    api_key: Optional[str] = None
    # Admin API key for privileged management endpoints.
    admin_api_key: Optional[str] = None
    # Public model name returned in OpenAI-compatible API responses.
    served_model_name: Optional[str] = None
    # Path used for server-side file uploads or temporary user artifacts.
    file_storage_path: Path = Path("mllm_storage")

    # -------------------------------------------------------------------------
    # Runtime and scheduling behavior
    # -------------------------------------------------------------------------
    # Fraction of total GPU memory reserved for static allocations
    # (primarily model weights + KV cache).
    mem_fraction_static: Optional[float] = None
    # Maximum number of requests concurrently executing in scheduler.
    max_running_requests: Optional[int] = None
    # Maximum queued requests waiting for execution.
    max_queued_requests: Optional[int] = None
    # Hard cap of total active tokens across all in-flight requests.
    max_total_tokens: Optional[int] = None
    # Prefill chunk size used to trade throughput vs memory pressure.
    chunked_prefill_size: Optional[int] = None
    # Upper bound for tokens accepted in a single prefill pass.
    max_prefill_tokens: int = 16384
    # Scheduling policy:
    # - "fcfs": first-come-first-served fairness.
    # - "lpm": longest-prefix-match style cache locality optimization.
    schedule_policy: Literal["fcfs", "lpm"] = "fcfs"
    # Conservative multiplier for scheduler admission decisions.
    # Values > 1.0 are safer for OOM avoidance but may reduce utilization.
    schedule_conservativeness: float = 1.0
    # Enable low-power sleep while idle to reduce background GPU usage.
    sleep_on_idle: bool = False
    # Stream partial output every N decode steps when streaming is enabled.
    stream_interval: int = 1
    # Enable token streaming in generation responses.
    stream_output: bool = True

    # -------------------------------------------------------------------------
    # Parallelism and distributed deployment
    # -------------------------------------------------------------------------
    # Tensor parallel size (intra-layer sharding).
    tp_size: int = 1
    # Data parallel size (replicated model workers).
    dp_size: int = 1
    # Expert parallel size for MoE-style models.
    ep_size: int = 1
    # Pipeline parallel size (inter-layer partitioning).
    pp_size: int = 1
    # Number of nodes participating in distributed serving.
    nnodes: int = 1
    # Rank of current node in multi-node topology.
    node_rank: int = 0
    # Torch distributed init address, e.g. "host:port".
    dist_init_addr: Optional[str] = None
    # Optional NCCL communication port override.
    nccl_port: Optional[int] = None
    # Timeout in seconds for distributed collectives.
    dist_timeout: Optional[int] = None
    # Base GPU index used for process-to-device mapping.
    base_gpu_id: int = 0
    # Step size between logical workers when assigning GPU IDs.
    gpu_id_step: int = 1

    # -------------------------------------------------------------------------
    # Backend and acceleration toggles
    # -------------------------------------------------------------------------
    # Attention kernel backend selection.
    attention_backend: Optional[str] = None
    # Sampling backend selection.
    sampling_backend: Optional[str] = None
    # Grammar-constrained decoding backend.
    grammar_backend: Optional[str] = None
    # Disable CUDA graph capture for debugging/compatibility.
    disable_cuda_graph: bool = False
    # Enable `torch.compile` acceleration path.
    enable_torch_compile: bool = False
    # Maximum batch size considered by `torch.compile` profiles.
    torch_compile_max_bs: int = 32
    # Enable deterministic inference behavior where possible.
    enable_deterministic_inference: bool = False
    # Random seed for reproducible sampling and initialization.
    random_seed: Optional[int] = None

    # -------------------------------------------------------------------------
    # Logging, metrics, and observability
    # -------------------------------------------------------------------------
    # Global log level for server components.
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    # HTTP access log level; if None, inherits global log level.
    log_level_http: Optional[str] = None
    # Log each request payload/metadata for debugging.
    log_requests: bool = False
    # Verbosity level for request logging, larger means more detail.
    log_requests_level: int = 2
    # Toggle built-in Prometheus/metrics endpoint.
    enable_metrics: bool = False
    # Include latency/time-cost summaries in logs.
    show_time_cost: bool = False
    # Optional OpenTelemetry traces endpoint ("host:port").
    otlp_traces_endpoint: str = "localhost:4317"
    # Enable tracing export to OTLP collector.
    enable_trace: bool = False

    # -------------------------------------------------------------------------
    # Feature switches and advanced decoding options
    # -------------------------------------------------------------------------
    # Enable LoRA adapter serving support.
    enable_lora: bool = False
    # Maximum number of LoRA adapters loaded simultaneously.
    max_loaded_loras: Optional[int] = None
    # Maximum LoRA adapters that can be mixed in one batch.
    max_loras_per_batch: int = 8
    # LoRA backend implementation.
    lora_backend: Literal["triton", "csgmv", "torch_native"] = "csgmv"
    # Enable multimodal processing pipeline.
    enable_multimodal: bool = False
    # Max concurrent multimodal tool calls.
    mm_max_concurrent_calls: int = 32
    # Timeout (seconds) for each multimodal call.
    mm_per_request_timeout: float = 10.0
    # Speculative decoding algorithm name (e.g. "eagle", "ngram").
    speculative_algorithm: Optional[str] = None
    # Draft model path used in speculative decoding.
    speculative_draft_model_path: Optional[Path] = None
    # Number of speculative steps per target decode iteration.
    speculative_num_steps: Optional[int] = None
    # Number of proposed draft tokens per speculation step.
    speculative_num_draft_tokens: Optional[int] = None

    # -------------------------------------------------------------------------
    # Internal bookkeeping (not usually set by users directly)
    # -------------------------------------------------------------------------
    # Additional arbitrary key-value options for forward compatibility.
    extra_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize defaults and validate constraints after dataclass initialization."""
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path
        if self.served_model_name is None:
            self.served_model_name = str(self.model_path)

        self._validate_basic_constraints()
        self._validate_parallelism_constraints()
        self._validate_scheduler_constraints()

    def _validate_basic_constraints(self) -> None:
        """Validate scalar ranges and common invariants."""
        if self.port <= 0 or self.port > 65535:
            raise ValueError("`port` must be in range [1, 65535].")
        if self.max_prefill_tokens <= 0:
            raise ValueError("`max_prefill_tokens` must be greater than 0.")
        if self.stream_interval <= 0:
            raise ValueError("`stream_interval` must be greater than 0.")
        if self.mem_fraction_static is not None and not (
            0.0 < self.mem_fraction_static < 1.0
        ):
            raise ValueError("`mem_fraction_static` must be in range (0.0, 1.0).")

    def _validate_parallelism_constraints(self) -> None:
        """Validate distributed and parallel topology settings."""
        for key, value in {
            "tp_size": self.tp_size,
            "dp_size": self.dp_size,
            "ep_size": self.ep_size,
            "pp_size": self.pp_size,
            "nnodes": self.nnodes,
        }.items():
            if value <= 0:
                raise ValueError(f"`{key}` must be greater than 0.")

        if self.node_rank < 0 or self.node_rank >= self.nnodes:
            raise ValueError("`node_rank` must satisfy 0 <= node_rank < nnodes.")

    def _validate_scheduler_constraints(self) -> None:
        """Validate scheduler-related soft limits."""
        if self.max_running_requests is not None and self.max_running_requests <= 0:
            raise ValueError("`max_running_requests` must be greater than 0 when set.")
        if self.max_queued_requests is not None and self.max_queued_requests < 0:
            raise ValueError("`max_queued_requests` must be >= 0 when set.")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("`max_total_tokens` must be greater than 0 when set.")
        if self.chunked_prefill_size is not None and self.chunked_prefill_size <= 0:
            raise ValueError("`chunked_prefill_size` must be greater than 0 when set.")
        if self.schedule_conservativeness <= 0:
            raise ValueError("`schedule_conservativeness` must be greater than 0.")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize config to a plain dictionary.

        Path values are converted to string for easier JSON/YAML serialization.
        """
        data = asdict(self)
        for key in [
            "model_path",
            "tokenizer_path",
            "download_dir",
            "file_storage_path",
            "speculative_draft_model_path",
        ]:
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data
