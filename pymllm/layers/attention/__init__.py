"""Attention layers and backends for pymllm."""

from pymllm.layers.attention.attention_backend import AttentionBackend
from pymllm.layers.attention.flashinfer_backend import (
    DecodeMetadata,
    FlashInferAttnBackend,
    PrefillMetadata,
    WrapperDispatch,
    should_use_tensor_core,
)
from pymllm.layers.attention.gdn_backend import GDNAttnBackend
from pymllm.layers.attention.hybrid_backend import HybridAttnBackend
from pymllm.layers.attention.radix_attention import AttentionType, RadixAttention
from pymllm.layers.attention.radix_linear_attention import RadixLinearAttention

__all__ = [
    # Base
    "AttentionBackend",
    # RadixAttention
    "AttentionType",
    "RadixAttention",
    # RadixLinearAttention (GDN)
    "RadixLinearAttention",
    # FlashInfer backend
    "FlashInferAttnBackend",
    "DecodeMetadata",
    "PrefillMetadata",
    "WrapperDispatch",
    "should_use_tensor_core",
    # GDN + Hybrid backends
    "GDNAttnBackend",
    "HybridAttnBackend",
]
