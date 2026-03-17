"""Abstract base class and shared data types for prefix cache implementations.

All concrete caches (:class:`RadixCache`, :class:`ChunkCache`,
:class:`MambaRadixCache`) inherit from :class:`BasePrefixCache` and share
the data classes defined here.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple, Union

import torch


# ======================================================================
# Hashing utilities
# ======================================================================


def hash_token_ids(
    token_ids: List[Union[int, Tuple[int, ...]]],
    prior_hash: Optional[str] = None,
) -> str:
    """SHA-256 hash of a token-id page with optional chain-hash.

    Each token is encoded as a 4-byte little-endian unsigned integer;
    tuples (bigram / EAGLE) hash each element in order.  When *prior_hash*
    is supplied the digest is seeded with the raw bytes of the previous
    hash, making the result position-aware.
    """
    hasher = hashlib.sha256()
    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))
    for t in token_ids:
        if isinstance(t, tuple):
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))
    return hasher.hexdigest()


def hash_to_int64(hex_str: str) -> int:
    """Convert a hex digest to a signed 64-bit integer (first 16 hex chars)."""
    val = int(hex_str[:16], 16)
    return val - (1 << 64) if val >= (1 << 63) else val


def hash_bytes(data: bytes) -> int:
    """SHA-256 -> unsigned 64-bit int.  Useful for multimodal embedding keys."""
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big", signed=False)


# ======================================================================
# Compound lookup key
# ======================================================================


class RadixKey:
    """Compound lookup key: token-id sequence + optional namespace tag.

    ``extra_key`` isolates independent namespaces so that sequences with
    identical leading tokens but different adapters / LoRA ids / multimodal
    context hashes never share prefix nodes.
    """

    __slots__ = ("token_ids", "extra_key")

    def __init__(
        self,
        token_ids: List[Union[int, Tuple[int, ...]]],
        extra_key: Optional[str] = None,
    ):
        self.token_ids = token_ids
        self.extra_key = extra_key

    def __len__(self) -> int:
        return len(self.token_ids)

    def __iter__(self) -> Iterator:
        return iter(self.token_ids)

    def __getitem__(self, idx: Union[int, slice]) -> RadixKey:
        if isinstance(idx, slice):
            return RadixKey(self.token_ids[idx], self.extra_key)
        return RadixKey([self.token_ids[idx]], self.extra_key)

    def __repr__(self) -> str:
        preview = self.token_ids[:10]
        tail = "..." if len(self.token_ids) > 10 else ""
        return f"RadixKey(extra={self.extra_key!r}, toks={preview}{tail})"


# ======================================================================
# Result data classes
# ======================================================================


@dataclass
class MatchResult:
    """Returned by :meth:`BasePrefixCache.match_prefix`."""

    indices: torch.Tensor
    last_node: Any = None
    prefix_len: int = 0
    # SSM / Mamba support
    mamba_branching_seqlen: Optional[int] = None


@dataclass
class InsertResult:
    """Returned by :meth:`BasePrefixCache.insert`."""

    prefix_len: int = 0
    last_node: Any = None
    # SSM / Mamba support: True when mamba state already existed in tree
    mamba_exist: bool = False


@dataclass
class EvictResult:
    """Returned by :meth:`BasePrefixCache.evict`."""

    full_evicted: int = 0
    swa_evicted: int = 0
    mamba_evicted: int = 0


# ======================================================================
# Abstract base class
# ======================================================================


class BasePrefixCache(ABC):
    """Abstract interface for all prefix cache implementations.

    Concrete implementations:

    * :class:`~pymllm.mem_cache.radix_cache.RadixCache` -- radix-tree with
      SWA tombstone support
    * :class:`~pymllm.mem_cache.chunk_cache.ChunkCache` -- no-op fallback
      (``disable_radix_cache=True``)
    * :class:`~pymllm.mem_cache.mamba_radix_cache.MambaRadixCache` -- radix-tree
      with independent Mamba/SSM state tracking
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all cached state and re-initialise."""
        ...

    @abstractmethod
    def match_prefix(self, key: RadixKey) -> MatchResult:
        """Find the longest cached prefix of *key*."""
        ...

    @abstractmethod
    def insert(
        self,
        key: RadixKey,
        value: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> InsertResult:
        """Insert *key*/*value* into the cache."""
        ...

    @abstractmethod
    def evict(self, num_tokens: int, swa_num_tokens: int = 0) -> EvictResult:
        """Evict tokens to free memory."""
        ...

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> Optional[Any]:
        """Lock *node* (and ancestors) to prevent eviction.

        Returns an opaque token (e.g. ``swa_boundary_id``) that must be
        passed back to :meth:`dec_lock_ref`.
        """
        ...

    @abstractmethod
    def dec_lock_ref(self, node: Any, **kwargs: Any) -> None:
        """Unlock *node* (and ancestors)."""
        ...

    # ------------------------------------------------------------------
    # Size queries (default implementations return 0)
    # ------------------------------------------------------------------

    def evictable_size(self) -> int:
        return 0

    def swa_evictable_size(self) -> int:
        return 0

    def protected_size(self) -> int:
        return 0

    def swa_protected_size(self) -> int:
        return 0

    def total_size(self) -> int:
        return 0
