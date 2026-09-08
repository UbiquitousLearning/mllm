"""Radix-tree KV cache with independent Mamba/SSM state tracking.

Extends :class:`~pymllm.mem_cache.radix_cache.RadixCache` with dual-tracked
state for hybrid models that combine full attention layers and SSM (Mamba /
GDN) layers.  Each tree node stores both:

- ``value``: KV-pool indices for full-attention layers
- ``mamba_value``: state-pool indices for SSM layers

The two pools have **independent reference counting and LRU eviction**:
Mamba state can be evicted more aggressively than full KV cache.

Reference: sglang ``MambaRadixCache``.
"""

from __future__ import annotations

import heapq
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from pymllm.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictResult,
    InsertResult,
    MatchResult,
    RadixKey,
)
from pymllm.mem_cache.radix_cache import (
    TreeNode as _BaseTreeNode,
    _child_key,
    _key_match,
    _next_node_id,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Mamba-aware tree node
# ======================================================================


class MambaTreeNode:
    """Tree node with dual KV + Mamba state tracking.

    Invariant: ``full_lock_ref >= mamba_lock_ref``.  If Mamba state is
    locked, full KV must also be locked; full KV alone can be locked
    without locking Mamba state.
    """

    __slots__ = (
        "children",
        "parent",
        "key",
        "value",
        "mamba_value",
        "full_lock_ref",
        "mamba_lock_ref",
        "last_access_time",
        "hit_count",
        "id",
        # LRU doubly-linked list pointers (full)
        "prev",
        "next",
        # LRU doubly-linked list pointers (mamba)
        "mamba_prev",
        "mamba_next",
    )

    def __init__(self) -> None:
        self.children: Dict[Any, MambaTreeNode] = defaultdict(MambaTreeNode)
        self.parent: Optional[MambaTreeNode] = None
        self.key: Optional[RadixKey] = None
        self.value: Optional[torch.Tensor] = None
        self.mamba_value: Optional[torch.Tensor] = None

        self.full_lock_ref: int = 0
        self.mamba_lock_ref: int = 0

        self.last_access_time: float = time.monotonic()
        self.hit_count: int = 0
        self.id: int = _next_node_id()

        # LRU list pointers
        self.prev: Optional[MambaTreeNode] = None
        self.next: Optional[MambaTreeNode] = None
        self.mamba_prev: Optional[MambaTreeNode] = None
        self.mamba_next: Optional[MambaTreeNode] = None

    @property
    def evicted(self) -> bool:
        return self.value is None

    @property
    def mamba_tombstone(self) -> bool:
        """Node has full KV but Mamba state was evicted."""
        return self.value is not None and self.mamba_value is None

    def __lt__(self, other: MambaTreeNode) -> bool:
        return self.last_access_time < other.last_access_time


# ======================================================================
# Doubly-linked LRU list
# ======================================================================


class LRUList:
    """Intrusive doubly-linked list for LRU ordering.

    Supports two modes via *mamba* flag: uses ``prev``/``next`` or
    ``mamba_prev``/``mamba_next`` pointers on :class:`MambaTreeNode`.
    """

    def __init__(self, mamba: bool = False):
        self.mamba = mamba
        if mamba:
            self._prv = "mamba_prev"
            self._nxt = "mamba_next"
            self._lock = "mamba_lock_ref"
        else:
            self._prv = "prev"
            self._nxt = "next"
            self._lock = "full_lock_ref"

        # Sentinel head (MRU side) and tail (LRU side)
        self.head = MambaTreeNode()
        self.tail = MambaTreeNode()
        setattr(self.head, self._nxt, self.tail)
        setattr(self.tail, self._prv, self.head)
        self._cache: Dict[int, MambaTreeNode] = {}

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, node: Optional[MambaTreeNode]) -> bool:
        return node is not None and node.id in self._cache

    # -- Mutations --------------------------------------------------------

    def insert_mru(self, node: MambaTreeNode) -> None:
        """Insert *node* at the MRU (head) position."""
        self._cache[node.id] = node
        self._add_after(self.head, node)

    def remove(self, node: MambaTreeNode) -> None:
        """Remove *node* from the list."""
        self._cache.pop(node.id, None)
        self._unlink(node)

    def touch_mru(self, node: MambaTreeNode) -> None:
        """Move an existing *node* to the MRU position."""
        if node.id not in self._cache:
            return
        self._unlink(node)
        self._add_after(self.head, node)

    def touch_node_and_parents_mru(
        self, node: MambaTreeNode, root: MambaTreeNode
    ) -> None:
        """Move *node* and all ancestors up to *root* to MRU.

        Child is more recently used than parent.
        """
        prev = self.head
        cur = node
        while cur != root:
            if cur.id in self._cache:
                if self.mamba and cur.mamba_value is None:
                    cur = cur.parent
                    continue
                self._unlink(cur)
                self._add_after(prev, cur)
                prev = cur
            cur = cur.parent

    # -- Queries ----------------------------------------------------------

    def get_lru_leaf_unlocked(self) -> Optional[MambaTreeNode]:
        """Return the LRU leaf node with lock_ref == 0, or ``None``."""
        x = getattr(self.tail, self._prv)
        while x != self.head:
            if getattr(x, self._lock) == 0 and len(x.children) == 0:
                return x
            x = getattr(x, self._prv)
        return None

    def get_lru_unlocked(self) -> Optional[MambaTreeNode]:
        """Return the LRU node with lock_ref == 0, or ``None``."""
        x = getattr(self.tail, self._prv)
        while x != self.head:
            if getattr(x, self._lock) == 0:
                return x
            x = getattr(x, self._prv)
        return None

    # -- Internal ---------------------------------------------------------

    def _add_after(self, old: MambaTreeNode, new: MambaTreeNode) -> None:
        nxt = getattr(old, self._nxt)
        setattr(new, self._prv, old)
        setattr(new, self._nxt, nxt)
        setattr(nxt, self._prv, new)
        setattr(old, self._nxt, new)

    def _unlink(self, node: MambaTreeNode) -> None:
        prv = getattr(node, self._prv)
        nxt = getattr(node, self._nxt)
        if prv is not None:
            setattr(prv, self._nxt, nxt)
        if nxt is not None:
            setattr(nxt, self._prv, prv)
        setattr(node, self._prv, None)
        setattr(node, self._nxt, None)


# ======================================================================
# MambaRadixCache
# ======================================================================


class MambaRadixCache(BasePrefixCache):
    """Radix tree with independent Mamba/SSM state tracking.

    Parameters
    ----------
    page_size:
        Number of tokens per KV-pool page.
    token_to_kv_pool_allocator:
        Pool allocator for full-attention KV indices.
    mamba_pool:
        Pool object for Mamba/SSM state.  Must support ``alloc_track_slot()``,
        ``free_track_slot(slot)``, ``copy_states(src, dst)``.
    on_node_evict:
        Optional callback invoked with node id on eviction.
    """

    def __init__(
        self,
        page_size: int = 1,
        token_to_kv_pool_allocator: Any = None,
        mamba_pool: Any = None,
        on_node_evict: Optional[Callable[[int], None]] = None,
    ):
        self.page_size = page_size
        self.pool = token_to_kv_pool_allocator
        self.mamba_pool = mamba_pool
        self.on_node_evict = on_node_evict

        if self.pool is not None and hasattr(self.pool, "device"):
            self.device = self.pool.device
        else:
            self.device = torch.device("cpu")

        # Dual LRU lists
        self.full_lru = LRUList(mamba=False)
        self.mamba_lru = LRUList(mamba=True)

        # Size counters
        self._full_evictable: int = 0
        self._full_protected: int = 0
        self._mamba_evictable: int = 0
        self._mamba_protected: int = 0

        self.reset()

    # ------------------------------------------------------------------
    # Size queries
    # ------------------------------------------------------------------

    def evictable_size(self) -> int:
        return self._full_evictable

    def protected_size(self) -> int:
        return self._full_protected

    def mamba_evictable_size(self) -> int:
        return self._mamba_evictable

    def mamba_protected_size(self) -> int:
        return self._mamba_protected

    def total_size(self) -> int:
        total = 0
        stack = [self.root_node]
        while stack:
            n = stack.pop()
            if n.value is not None:
                total += len(n.value)
            stack.extend(c for c in n.children.values() if not c.evicted)
        return total

    # ------------------------------------------------------------------
    # BasePrefixCache interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.root_node = MambaTreeNode()
        self.root_node.key = RadixKey([])
        self.root_node.value = torch.tensor([], dtype=torch.int64)
        self.root_node.mamba_value = torch.tensor([], dtype=torch.int64)
        self.root_node.full_lock_ref = 1
        self.root_node.mamba_lock_ref = 1
        self._full_evictable = 0
        self._full_protected = 0
        self._mamba_evictable = 0
        self._mamba_protected = 0
        self.full_lru = LRUList(mamba=False)
        self.mamba_lru = LRUList(mamba=True)

    def match_prefix(self, key: RadixKey) -> MatchResult:
        """Find longest cached prefix.  Also returns ``mamba_branching_seqlen``."""
        empty = MatchResult(
            indices=torch.empty(0, dtype=torch.int64, device=self.device),
            last_node=self.root_node,
        )
        if len(key) == 0:
            return empty

        key = self._page_align_key(key)
        if len(key) == 0:
            return empty

        node = self.root_node
        values: List[torch.Tensor] = []
        mamba_branching_seqlen: Optional[int] = None
        total_matched = 0

        while len(key) > 0:
            ck = _child_key(key, self.page_size)
            if ck not in node.children:
                break
            child = node.children[ck]
            child.hit_count += 1
            plen = _key_match(child.key, key, self.page_size)

            if plen < len(child.key):
                new_node = self._split_node(child.key, child, plen)
                values.append(new_node.value)
                # Track mamba branching point
                if mamba_branching_seqlen is None and new_node.mamba_tombstone:
                    mamba_branching_seqlen = total_matched
                total_matched += len(new_node.value)
                node = new_node
                break

            values.append(child.value)
            if mamba_branching_seqlen is None and child.mamba_tombstone:
                mamba_branching_seqlen = total_matched
            total_matched += len(child.value)
            node = child
            key = key[plen:]

        # Update LRU for matched path
        self.full_lru.touch_node_and_parents_mru(node, self.root_node)
        self.mamba_lru.touch_node_and_parents_mru(node, self.root_node)

        cat = (
            torch.cat(values)
            if values
            else torch.empty(0, dtype=torch.int64, device=self.device)
        )
        return MatchResult(
            indices=cat,
            last_node=node,
            prefix_len=len(cat),
            mamba_branching_seqlen=mamba_branching_seqlen,
        )

    def insert(
        self,
        key: RadixKey,
        value: Optional[torch.Tensor] = None,
        *,
        mamba_value: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> InsertResult:
        """Insert with both full KV and Mamba state values."""
        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)

        if len(key) == 0:
            return InsertResult()

        node = self.root_node
        total_prefix = 0
        mamba_exist = False

        ck = _child_key(key, self.page_size)
        while len(key) > 0 and ck in node.children:
            node = node.children[ck]
            plen = _key_match(node.key, key, self.page_size)
            total_prefix += plen
            key = key[plen:]
            value = value[plen:]

            if plen < len(node.key):
                node = self._split_node(node.key, node, plen)

            # Check if mamba state already exists
            if node.mamba_value is not None:
                mamba_exist = True

            if len(key) > 0:
                ck = _child_key(key, self.page_size)

        if len(key) > 0:
            new_leaf = self._add_leaf(node, key, value, mamba_value=mamba_value)
            node = new_leaf
        elif mamba_value is not None and node.mamba_value is None:
            # Existing node gains mamba state (un-tombstone)
            node.mamba_value = mamba_value.clone()
            self.mamba_lru.insert_mru(node)
            self._mamba_evictable += len(node.value)

        return InsertResult(
            prefix_len=total_prefix, last_node=node, mamba_exist=mamba_exist
        )

    def evict(self, num_tokens: int, swa_num_tokens: int = 0) -> EvictResult:
        """Evict full KV and/or Mamba state tokens.

        Phase 1: Evict full KV leaves (frees both KV and Mamba state).
        Phase 2: Evict Mamba state from internal nodes (tombstone mamba).
        """
        full_evicted = 0
        mamba_evicted = 0

        # Phase 1: full leaf eviction
        if num_tokens > 0:
            while full_evicted < num_tokens:
                node = self.full_lru.get_lru_leaf_unlocked()
                if node is None:
                    break
                n = len(node.value)
                self._free_full_indices(node.value)
                if node.mamba_value is not None:
                    self._free_mamba_value(node.mamba_value)
                    mamba_evicted += n
                full_evicted += n
                self._delete_leaf(node)

                # Cascade: parent may become evictable leaf
                p = node.parent
                if (
                    p is not None
                    and p != self.root_node
                    and len(p.children) == 0
                    and p.full_lock_ref == 0
                ):
                    # Will be picked up in next iteration via LRU
                    pass

        # Phase 2: mamba-only tombstone eviction
        target_mamba = swa_num_tokens
        if target_mamba > 0 and mamba_evicted < target_mamba:
            while mamba_evicted < target_mamba:
                node = self.mamba_lru.get_lru_unlocked()
                if node is None:
                    break
                if node.mamba_value is None:
                    continue
                n = len(node.mamba_value)
                self._free_mamba_value(node.mamba_value)
                node.mamba_value = None
                self.mamba_lru.remove(node)
                self._mamba_evictable -= n
                mamba_evicted += n

        return EvictResult(
            full_evicted=full_evicted, mamba_evicted=mamba_evicted
        )

    def inc_lock_ref(self, node: MambaTreeNode) -> Optional[Any]:
        """Lock full KV and Mamba state from *node* to root.

        Full lock propagates up to root.  Mamba lock only applies to
        the node itself (not ancestors).
        """
        if node is None:
            return None

        # Lock mamba on the node itself
        if node.mamba_value is not None:
            if node.mamba_lock_ref == 0 and node in self.mamba_lru:
                self._mamba_evictable -= len(node.mamba_value)
                self._mamba_protected += len(node.mamba_value)
            node.mamba_lock_ref += 1

        # Lock full KV up to root
        cur = node
        while cur != self.root_node:
            if cur.full_lock_ref == 0:
                self._full_evictable -= len(cur.key)
                self._full_protected += len(cur.key)
            cur.full_lock_ref += 1
            cur = cur.parent
        return None

    def dec_lock_ref(self, node: MambaTreeNode, **kwargs: Any) -> None:
        """Unlock full KV and Mamba state."""
        if node is None:
            return

        # Unlock mamba on the node itself
        if node.mamba_lock_ref > 0:
            node.mamba_lock_ref -= 1
            if node.mamba_lock_ref == 0 and node.mamba_value is not None:
                self._mamba_evictable += len(node.mamba_value)
                self._mamba_protected -= len(node.mamba_value)

        # Unlock full KV up to root
        cur = node
        while cur != self.root_node:
            if cur.full_lock_ref == 1:
                self._full_evictable += len(cur.key)
                self._full_protected -= len(cur.key)
            cur.full_lock_ref -= 1
            cur = cur.parent

    # ------------------------------------------------------------------
    # Internal: tree manipulation
    # ------------------------------------------------------------------

    def _add_leaf(
        self,
        parent: MambaTreeNode,
        key: RadixKey,
        value: torch.Tensor,
        mamba_value: Optional[torch.Tensor] = None,
    ) -> MambaTreeNode:
        # Note: we intentionally do NOT subtract parent's tokens from
        # _full_evictable when a leaf gains its first child.  Internal
        # nodes are still reclaimable via cascade eviction (evict children
        # first, then the childless parent cascades).  Subtracting here
        # would break the invariant that evictable + protected == total
        # tree tokens.  See RadixCache._add_leaf for full rationale.

        new_node = MambaTreeNode()
        new_node.parent = parent
        new_node.key = key
        new_node.value = value.clone()
        parent.children[_child_key(key, self.page_size)] = new_node

        # Track in full LRU
        self.full_lru.insert_mru(new_node)
        self._full_evictable += len(key)

        # Track mamba state if provided
        if mamba_value is not None:
            new_node.mamba_value = mamba_value.clone()
            self.mamba_lru.insert_mru(new_node)
            self._mamba_evictable += len(key)

        return new_node

    def _split_node(
        self, key: RadixKey, child: MambaTreeNode, split_len: int
    ) -> MambaTreeNode:
        """Split *child* at *split_len*, returning the new parent node."""
        new_node = MambaTreeNode()
        new_node.children[_child_key(key[split_len:], self.page_size)] = child
        new_node.parent = child.parent
        new_node.full_lock_ref = child.full_lock_ref
        new_node.mamba_lock_ref = child.mamba_lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()

        # Split mamba value
        if child.mamba_value is not None:
            new_node.mamba_value = child.mamba_value[:split_len].clone()
            child.mamba_value = child.mamba_value[split_len:].clone()

        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[_child_key(key, self.page_size)] = new_node

        # Update LRU lists: insert new_node, keep child
        self.full_lru.insert_mru(new_node)
        if new_node.mamba_value is not None:
            self.mamba_lru.insert_mru(new_node)

        return new_node

    def _delete_leaf(self, node: MambaTreeNode) -> None:
        ck = _child_key(node.key, self.page_size)
        node.parent.children.pop(ck, None)

        # Remove from LRU lists
        if node in self.full_lru:
            self.full_lru.remove(node)
        self._full_evictable -= len(node.key)

        if node.mamba_value is not None and node in self.mamba_lru:
            self.mamba_lru.remove(node)
            self._mamba_evictable -= len(node.key)

        node.value = None
        node.mamba_value = None

        if self.on_node_evict is not None:
            self.on_node_evict(node.id)

    # ------------------------------------------------------------------
    # Internal: memory management
    # ------------------------------------------------------------------

    def _free_full_indices(self, indices: torch.Tensor) -> None:
        if self.pool is not None and len(indices) > 0:
            self.pool.free(indices)

    def _free_mamba_value(self, mamba_value: torch.Tensor) -> None:
        if self.mamba_pool is not None and len(mamba_value) > 0:
            for idx in mamba_value.tolist():
                self.mamba_pool.free_track_slot(int(idx))

    def _page_align_key(self, key: RadixKey) -> RadixKey:
        if self.page_size == 1:
            return key
        aligned = len(key) // self.page_size * self.page_size
        return key[:aligned]

    def pretty_print(self) -> None:
        """Print the tree structure to stdout."""
        self._print_helper(self.root_node, 0)
        print(
            f"total={self.total_size()}  "
            f"full_evictable={self._full_evictable}  "
            f"mamba_evictable={self._mamba_evictable}"
        )

    def _print_helper(self, node: MambaTreeNode, indent: int) -> None:
        stack = [(node, indent)]
        while stack:
            n, ind = stack.pop()
            toks = n.key.token_ids[:10] if n.key else []
            klen = len(n.key) if n.key else 0
            has_mamba = n.mamba_value is not None
            print(
                f"{'  ' * ind}[{klen}] {toks} "
                f"full_lock={n.full_lock_ref} mamba_lock={n.mamba_lock_ref} "
                f"mamba={'Y' if has_mamba else 'N'}"
            )
            for c in n.children.values():
                stack.append((c, ind + 1))
