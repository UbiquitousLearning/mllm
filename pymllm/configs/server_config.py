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
    max_prefill_tokens: int = None
    schedule_policy: Literal["auto", "fcfs"] = "fcfs"
    schedule_conservativeness: float = 1.0
    sleep_on_idle: bool = False
    stream_interval: int = 1
    stream_output: bool = True

    # --------------------------------------------------------------------- #
    # Threads
    # --------------------------------------------------------------------- #
    enable_disk_io_async: bool = False
    disk_io_async_thread_count: int = 1

    # --------------------------------------------------------------------- #
    # Device
    # --------------------------------------------------------------------- #
    base_gpu_id: int = 0

    # --------------------------------------------------------------------- #
    # Backend / acceleration
    # --------------------------------------------------------------------- #
    attention_backend: Literal["auto", "flashinfer"] = "auto"
    sampling_backend: Optional[str] = None
    disable_cuda_graph: bool = False
    enable_torch_compile: bool = True
    torch_compile_max_bs: int = 32
    random_seed: Optional[int] = 42

    # --------------------------------------------------------------------- #
    # Logging and observability
    # --------------------------------------------------------------------- #
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
    enable_metrics: bool = False
    show_time_cost: bool = False

    # --------------------------------------------------------------------- #
    # Feature switches
    # --------------------------------------------------------------------- #
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
        if self.schedule_conservativeness <= 0:
            raise ValueError("`schedule_conservativeness` must be > 0.")
