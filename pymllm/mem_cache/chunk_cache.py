"""No-op prefix cache used when ``disable_radix_cache=True``.

Every request is fully computed from scratch -- no prefix sharing, no
tree structure, no eviction logic.  This is the simplest possible
:class:`~pymllm.mem_cache.base_prefix_cache.BasePrefixCache` implementation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from pymllm.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictResult,
    InsertResult,
    MatchResult,
    RadixKey,
)


class ChunkCache(BasePrefixCache):
    """No-op prefix cache: no prefix sharing, no eviction.

    When the radix cache is disabled, this class replaces it so that
    the rest of the system can call the same interface without branching.

    Parameters
    ----------
    token_to_kv_pool_allocator:
        Pool allocator used to free KV indices on request completion.
    device:
        Device for empty tensors returned by :meth:`match_prefix`.
    """

    def __init__(
        self,
        token_to_kv_pool_allocator: Any = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.pool = token_to_kv_pool_allocator
        self.device = device

    def reset(self) -> None:
        pass

    def match_prefix(self, key: RadixKey) -> MatchResult:
        """Always returns an empty match (no prefix sharing)."""
        return MatchResult(
            indices=torch.empty(0, dtype=torch.int64, device=self.device),
            last_node=None,
        )

    def insert(
        self,
        key: RadixKey,
        value: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> InsertResult:
        """No-op: nothing is cached."""
        return InsertResult()

    def evict(self, num_tokens: int, swa_num_tokens: int = 0) -> EvictResult:
        """No-op: nothing to evict."""
        return EvictResult()

    def inc_lock_ref(self, node: Any) -> Optional[Any]:
        """No-op: nothing to lock."""
        return None

    def dec_lock_ref(self, node: Any, **kwargs: Any) -> None:
        """No-op: nothing to unlock."""
        pass
