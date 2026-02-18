"""Global configuration singleton with all server, model and runtime configs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig


@dataclass
class ModelConfig:
    """Model-specific configuration parsed from HF config.
    
    This is a lightweight wrapper around HuggingFace config with
    additional derived fields for efficiency.
    """
    # Original HF config (populated after loading)
    hf_config: Optional[Any] = field(default=None, repr=False)
    hf_text_config: Optional[Any] = field(default=None, repr=False)
    
    # Model architecture
    model_type: str = "unknown"
    architectures: list[str] = field(default_factory=list)
    
    # Dimensions
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: Optional[int] = None
    intermediate_size: int = 0
    vocab_size: int = 0
    
    # Context length
    max_position_embeddings: int = 0
    context_length: int = 0  # effective context length
    
    # Normalization
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    
    # RoPE
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Quantization
    quantization: Optional[str] = None
    
    def __post_init__(self):
        """Set default kv heads if not specified."""
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


@dataclass  
class RuntimeConfig:
    """Runtime state that changes during execution."""
    
    # Distributed state
    tp_rank: int = 0
    tp_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    pp_rank: int = 0
    pp_size: int = 1
    world_rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    
    # Device
    device: str = "cuda"
    
    # Memory pools
    max_num_seqs: int = 0
    max_model_len: int = 0
    
    # Scheduler state (mutable during runtime)
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    num_swapped_reqs: int = 0


@dataclass
class CacheConfig:
    """KV cache configuration."""
    
    block_size: int = 16
    num_gpu_blocks: int = 0
    num_cpu_blocks: int = 0
    
    # Cache dtype
    cache_dtype: Literal["auto", "float16", "bfloat16", "fp8_e4m3", "fp8_e5m2"] = "auto"
    
    # Sliding window
    sliding_window: Optional[int] = None
    
    # Prefix caching
    enable_prefix_caching: bool = False


@dataclass
class GlobalConfig:
    """Global configuration singleton containing all configs.
    
    This is the single source of truth for all configuration in pymllm.
    It aggregates ServerConfig, ModelConfig, RuntimeConfig, and CacheConfig.
    
    Usage:
        >>> from pymllm.configs import get_global_config
        >>> config = get_global_config()
        >>> 
        >>> # Access server config
        >>> config.server.model_path
        >>> config.server.tp_size
        >>> 
        >>> # Access model config
        >>> config.model.hidden_size
        >>> config.model.vocab_size
        >>> 
        >>> # Access runtime config (mutable)
        >>> config.runtime.tp_rank
        >>> config.runtime.device
        >>> 
        >>> # Access cache config
        >>> config.cache.block_size
        >>> 
        >>> # Update with new server config
        >>> config.load_server_config(server_config)
        >>> 
        >>> # Update with HF model config
        >>> config.load_hf_config(hf_config)
    """
    
    # Sub-configs
    server: "ServerConfig" = field(default=None, repr=False)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    
    # Additional metadata
    _initialized: bool = field(default=False, repr=False)
    
    def __new__(cls):
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __post_init__(self):
        # Lazy import to avoid circular dependency
        if self.server is None:
            from pymllm.configs.server_config import ServerConfig
            self.server = ServerConfig(
                model_path=Path("."),  # placeholder
            )
    
    @classmethod
    def get_instance(cls) -> "GlobalConfig":
        """Get the singleton instance."""
        if not hasattr(cls, '_instance') or cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def load_server_config(self, server_config: "ServerConfig") -> None:
        """Load server configuration and sync related fields."""
        self.server = server_config
        
        # Sync tp/dp/pp sizes to runtime
        self.runtime.tp_size = server_config.tp_size
        self.runtime.dp_size = server_config.dp_size
        self.runtime.pp_size = server_config.pp_size
        self.runtime.device = "cuda" if server_config.base_gpu_id >= 0 else "cpu"
        
        self._initialized = True
    
    def load_hf_config(self, hf_config: "PretrainedConfig") -> None:
        """Load HuggingFace model configuration."""
        from transformers import PretrainedConfig
        
        # Store original
        self.model.hf_config = hf_config
        
        # Get text config (for multimodal models)
        if hasattr(hf_config, "text_config"):
            self.model.hf_text_config = hf_config.text_config
            text_config = hf_config.text_config
        else:
            text_config = hf_config
            self.model.hf_text_config = hf_config
        
        # Extract fields
        self.model.model_type = getattr(text_config, "model_type", "unknown")
        self.model.architectures = getattr(text_config, "architectures", [])
        
        self.model.hidden_size = getattr(text_config, "hidden_size", 0)
        self.model.num_hidden_layers = getattr(text_config, "num_hidden_layers", 0)
        self.model.num_attention_heads = getattr(text_config, "num_attention_heads", 0)
        self.model.num_key_value_heads = getattr(text_config, "num_key_value_heads", None)
        self.model.intermediate_size = getattr(text_config, "intermediate_size", 0)
        self.model.vocab_size = getattr(text_config, "vocab_size", 0)
        
        # Context length
        self.model.max_position_embeddings = getattr(
            text_config, "max_position_embeddings", 0
        )
        self.model.context_length = self._get_context_length(text_config)
        
        # Normalization
        self.model.rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-6)
        self.model.tie_word_embeddings = getattr(
            text_config, "tie_word_embeddings", False
        )
        
        # RoPE
        self.model.rope_theta = getattr(text_config, "rope_theta", 10000.0)
        self.model.rope_scaling = getattr(text_config, "rope_scaling", None)
        
        # Sync to cache config
        self.cache.sliding_window = getattr(text_config, "sliding_window", None)
    
    def _get_context_length(self, config: "PretrainedConfig") -> int:
        """Extract effective context length from config."""
        # Try various fields
        for key in ["max_position_embeddings", "n_positions", "seq_length"]:
            if hasattr(config, key):
                value = getattr(config, key)
                if isinstance(value, int) and value > 0:
                    return value
        return 2048  # default
    
    def update_runtime(self, **kwargs) -> None:
        """Update runtime configuration."""
        for key, value in kwargs.items():
            if hasattr(self.runtime, key):
                setattr(self.runtime, key, value)
            else:
                raise AttributeError(f"RuntimeConfig has no attribute '{key}'")
    
    def update_cache(self, **kwargs) -> None:
        """Update cache configuration."""
        for key, value in kwargs.items():
            if hasattr(self.cache, key):
                setattr(self.cache, key, value)
            else:
                raise AttributeError(f"CacheConfig has no attribute '{key}'")
    
    def temp(self, **kwargs):
        """Context manager for temporary config changes.
        
        Usage:
            # Modify runtime config temporarily
            with config.temp(runtime=config.runtime):
                config.runtime.tp_size = 2
                # ... do something with tp_size=2
            # runtime restored to original values
        """
        return _TempGlobalConfig(self, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize all configs to dictionary."""
        return {
            "server": self.server.to_dict() if self.server else {},
            "model": self._model_to_dict(),
            "runtime": self._runtime_to_dict(),
            "cache": self._cache_to_dict(),
        }
    
    def _model_to_dict(self) -> Dict[str, Any]:
        """Convert model config to dict."""
        return {
            "model_type": self.model.model_type,
            "architectures": self.model.architectures,
            "hidden_size": self.model.hidden_size,
            "num_hidden_layers": self.model.num_hidden_layers,
            "num_attention_heads": self.model.num_attention_heads,
            "num_key_value_heads": self.model.num_key_value_heads,
            "intermediate_size": self.model.intermediate_size,
            "vocab_size": self.model.vocab_size,
            "context_length": self.model.context_length,
        }
    
    def _runtime_to_dict(self) -> Dict[str, Any]:
        """Convert runtime config to dict."""
        return {
            "tp_rank": self.runtime.tp_rank,
            "tp_size": self.runtime.tp_size,
            "world_rank": self.runtime.world_rank,
            "world_size": self.runtime.world_size,
            "device": self.runtime.device,
        }
    
    def _cache_to_dict(self) -> Dict[str, Any]:
        """Convert cache config to dict."""
        return {
            "block_size": self.cache.block_size,
            "num_gpu_blocks": self.cache.num_gpu_blocks,
            "cache_dtype": self.cache.cache_dtype,
        }


class _TempGlobalConfig:
    """Context manager for temporary global config changes.
    
    Supports nested keys like "runtime.tp_size" to modify sub-configs.
    """
    
    def __init__(self, config: GlobalConfig, **kwargs):
        self.config = config
        self.temp_values = kwargs
        self.old_values = {}
    
    def _get_nested_attr(self, key: str):
        """Get attribute, supporting dot notation for nested access."""
        if "." in key:
            parts = key.split(".")
            obj = self.config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            return getattr(obj, parts[-1])
        return getattr(self.config, key)
    
    def _set_nested_attr(self, key: str, value):
        """Set attribute, supporting dot notation for nested access."""
        if "." in key:
            parts = key.split(".")
            obj = self.config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(self.config, key, value)
    
    def __enter__(self):
        for key, value in self.temp_values.items():
            self.old_values[key] = self._get_nested_attr(key)
            self._set_nested_attr(key, value)
        return self.config
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, value in self.old_values.items():
            self._set_nested_attr(key, value)
        return False


# Convenience function
def get_global_config() -> GlobalConfig:
    """Get the global config singleton instance."""
    return GlobalConfig.get_instance()
