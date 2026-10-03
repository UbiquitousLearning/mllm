"""Radix-tree KV cache with SWA and multimodal support.

Supports:
    - Multi-batch serving on a single GPU
    - Sliding Window Attention (SWA) via tombstone mechanism
    - Multimodal namespace isolation via ``extra_key``
    - SHA256 position-aware hashing
    - Page-aligned operations (page_size >= 1)
    - LRU leaf eviction
"""

from __future__ import annotations

import heapq
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from pymllm.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    EvictResult,
    InsertResult,
    MatchResult,
    RadixKey,
    hash_token_ids,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Tree node
# ======================================================================

_node_counter: int = 0


def _next_node_id() -> int:
    global _node_counter
    _node_counter += 1
    return _node_counter


class TreeNode:
    """A single node in the radix tree.

    ``value`` holds a 1-D ``int64`` tensor of KV-pool indices (one per token
    in ``key``).  When the node has been evicted, ``value`` is ``None``.
    """

    __slots__ = (
        "children",
        "parent",
        "key",
        "value",
        "lock_ref",
        "swa_lock_ref",
        "swa_tombstone",
        "swa_boundary_id",
        "last_access_time",
        "hit_count",
        "hash_values",
        "id",
    )

    def __init__(self) -> None:
        self.children: Dict[Any, TreeNode] = defaultdict(TreeNode)
        self.parent: Optional[TreeNode] = None
        self.key: Optional[RadixKey] = None
        self.value: Optional[torch.Tensor] = None

        self.lock_ref: int = 0
        self.swa_lock_ref: int = 0
        self.swa_tombstone: bool = False
        self.swa_boundary_id: Optional[int] = None

        self.last_access_time: float = time.monotonic()
        self.hit_count: int = 0
        self.hash_values: Optional[List[str]] = None
        self.id: int = _next_node_id()

    @property
    def evicted(self) -> bool:
        return self.value is None

    def __lt__(self, other: TreeNode) -> bool:
        return self.last_access_time < other.last_access_time


# ======================================================================
# Helper functions
# ======================================================================


def _key_match(key0: RadixKey, key1: RadixKey, page_size: int) -> int:
    """Return the length of the common prefix (page-aligned when *page_size* > 1)."""
    if key0.extra_key != key1.extra_key:
        return 0
    if page_size == 1:
        i = 0
        for a, b in zip(key0.token_ids, key1.token_ids):
            if a != b:
                break
            i += 1
        return i
    min_len = min(len(key0), len(key1))
    i = 0
    while i < min_len:
        if key0.token_ids[i : i + page_size] != key1.token_ids[i : i + page_size]:
            break
        i += page_size
    return i


def _child_key(key: RadixKey, page_size: int) -> Any:
    """Derive the dict key used in ``node.children``."""
    plain = key.token_ids[0] if page_size == 1 else tuple(key.token_ids[:page_size])
    return (key.extra_key, plain) if key.extra_key is not None else plain


# ======================================================================
# RadixCache
# ======================================================================


class RadixCache(BasePrefixCache):
    """Radix tree for KV-cache prefix sharing.

    Parameters
    ----------
    page_size:
        Number of tokens per KV-pool page.  Keys and values are aligned to
        this granularity.
    sliding_window_size:
        If set, enables SWA mode.  The cache tracks which nodes have had
        their SWA KV freed (tombstoned) and constrains prefix matching
        so that the sliding-window invariant is maintained.
    token_to_kv_pool_allocator:
        Optional pool allocator with ``free(indices)`` (and ``free_swa`` for
        SWA mode).  When *None*, index tensors are simply discarded.
    on_node_evict:
        Optional callback invoked with the node id when a node is evicted.
    """

    def __init__(
        self,
        page_size: int = 1,
        sliding_window_size: Optional[int] = None,
        token_to_kv_pool_allocator: Any = None,
        on_node_evict: Optional[Callable[[int], None]] = None,
    ):
        self.page_size = page_size
        self.sliding_window_size = sliding_window_size
        self.pool = token_to_kv_pool_allocator
        self.on_node_evict = on_node_evict

        if self.pool is not None and hasattr(self.pool, "device"):
            self.device = self.pool.device
        else:
            self.device = torch.device("cpu")

        self._swa_boundary_counter: int = 0
        self.reset()

    @property
    def supports_swa(self) -> bool:
        return self.sliding_window_size is not None

    # ------------------------------------------------------------------
    # Size queries
    # ------------------------------------------------------------------

    def evictable_size(self) -> int:
        return self._evictable_size

    def swa_evictable_size(self) -> int:
        return self._swa_evictable_size

    def protected_size(self) -> int:
        return self._protected_size

    def swa_protected_size(self) -> int:
        return self._swa_protected_size

    def total_size(self) -> int:
        """Total number of cached tokens (including tombstoned)."""
        total = 0
        stack: List[TreeNode] = [self.root_node]
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
        """Clear all cached state and re-initialise the root node."""
        self.root_node = TreeNode()
        self.root_node.key = RadixKey([])
        self.root_node.value = torch.tensor([], dtype=torch.int64)
        self.root_node.lock_ref = 1
        self.root_node.swa_lock_ref = 1
        self._evictable_size: int = 0
        self._swa_evictable_size: int = 0
        self._protected_size: int = 0
        self._swa_protected_size: int = 0

    def match_prefix(self, key: RadixKey) -> MatchResult:
        """Find the longest cached prefix of *key*.

        For SWA mode the match is further constrained: the path from the
        returned ``last_node`` to root must have at least
        ``sliding_window_size`` non-tombstone tokens (or be entirely
        tombstone-free back to root).

        Accessing a prefix refreshes LRU timestamps along the matched path.
        """
        empty = MatchResult(
            indices=torch.empty(0, dtype=torch.int64, device=self.device),
            last_node=self.root_node,
        )
        if len(key) == 0:
            return empty

        key = self._page_align_key(key)
        if len(key) == 0:
            return empty

        if self.supports_swa:
            values, last_node, best_count = self._match_swa(key)
            values = values[:best_count]
        else:
            values, last_node = self._match_normal(key)

        cat = (
            torch.cat(values)
            if values
            else torch.empty(0, dtype=torch.int64, device=self.device)
        )
        return MatchResult(indices=cat, last_node=last_node, prefix_len=len(cat))

    def insert(
        self,
        key: RadixKey,
        value: Optional[torch.Tensor] = None,
        *,
        prev_prefix_len: int = 0,
        swa_evicted_seqlen: int = 0,
        **kwargs: Any,
    ) -> InsertResult:
        """Insert *key*/*value* into the tree.

        Returns how many leading tokens were already present (the prefix
        length).  The caller is responsible for freeing duplicate KV indices
        in the range ``[cache_protected_len, prefix_len)``.

        Parameters
        ----------
        prev_prefix_len:
            (SWA mode) tokens before this offset are already protected and
            should not have their values overwritten.
        swa_evicted_seqlen:
            (SWA mode) the sequence length up to which SWA KV has been
            previously evicted.  Used to decide whether a tombstoned node can
            be un-tombstoned with the incoming value.
        """
        if value is None:
            value = torch.tensor(key.token_ids, dtype=torch.int64)
        if self.supports_swa:
            plen = self._insert_swa(
                self.root_node, key, value, prev_prefix_len, swa_evicted_seqlen
            )
            return InsertResult(prefix_len=plen)
        else:
            plen, last_node = self._insert_normal(self.root_node, key, value)
            return InsertResult(prefix_len=plen, last_node=last_node)

    def evict(self, num_tokens: int, swa_num_tokens: int = 0) -> EvictResult:
        """Evict up to *num_tokens* (full) and *swa_num_tokens* (SWA) tokens.

        Full eviction removes leaf nodes entirely; SWA eviction tombstones
        internal nodes (freeing SWA KV but retaining full-attn KV).
        """
        full_evicted = 0
        swa_evicted = 0

        # Phase 1: full leaf eviction
        if num_tokens > 0:
            leaves = self._collect_evictable_leaves()
            heap: List[Tuple[float, TreeNode]] = [
                (n.last_access_time, n) for n in leaves
            ]
            heapq.heapify(heap)

            while full_evicted < num_tokens and heap:
                _, node = heapq.heappop(heap)
                if node.evicted or node.lock_ref > 0:
                    continue
                n = len(node.value)
                self._free_indices(node.value)
                full_evicted += n
                swa_evicted += n
                self._delete_leaf(node)

                p = node.parent
                if (
                    p is not None
                    and p != self.root_node
                    and len(p.children) == 0
                    and p.lock_ref == 0
                ):
                    if self.supports_swa and p.swa_tombstone:
                        self._free_indices(p.value)
                        full_evicted += len(p.value)
                        self._delete_leaf(p)
                    else:
                        heapq.heappush(heap, (p.last_access_time, p))

        # Phase 2: SWA tombstone eviction (internal nodes)
        if self.supports_swa and swa_evicted < swa_num_tokens:
            candidates = self._collect_swa_evictable()
            heap2: List[Tuple[float, TreeNode]] = [
                (n.last_access_time, n) for n in candidates
            ]
            heapq.heapify(heap2)

            while swa_evicted < swa_num_tokens and heap2:
                _, node = heapq.heappop(heap2)
                if node.swa_tombstone or node.swa_lock_ref > 0 or node.evicted:
                    continue
                n = len(node.value)
                if len(node.children) == 0 and node.lock_ref == 0:
                    self._free_indices(node.value)
                    full_evicted += n
                    swa_evicted += n
                    self._delete_leaf(node)
                elif len(node.children) > 0:
                    self._free_swa_indices(node.value)
                    swa_evicted += n
                    self._tombstone_node(node)

        return EvictResult(full_evicted=full_evicted, swa_evicted=swa_evicted)

    def inc_lock_ref(self, node: TreeNode) -> Optional[int]:
        """Lock nodes from *node* up to root (prevents eviction).

        Returns ``swa_boundary_id`` that must be passed back to
        :meth:`dec_lock_ref`.  In non-SWA mode, returns ``None``.
        """
        if node is None:
            return None

        swa_locked = 0
        swa_boundary_id: Optional[int] = None
        cur = node
        while cur != self.root_node:
            if cur.lock_ref == 0:
                self._evictable_size -= len(cur.key)
                self._protected_size += len(cur.key)
            cur.lock_ref += 1

            if (
                self.supports_swa
                and swa_locked < self.sliding_window_size
                and not cur.swa_tombstone
            ):
                if cur.swa_lock_ref == 0:
                    self._swa_evictable_size -= len(cur.key)
                    self._swa_protected_size += len(cur.key)
                cur.swa_lock_ref += 1
                swa_locked += len(cur.key)
                if swa_locked >= self.sliding_window_size:
                    if cur.swa_boundary_id is None:
                        self._swa_boundary_counter += 1
                        cur.swa_boundary_id = self._swa_boundary_counter
                    swa_boundary_id = cur.swa_boundary_id

            cur = cur.parent
        return swa_boundary_id

    def dec_lock_ref(
        self, node: TreeNode, swa_boundary_id: Optional[int] = None, **kwargs: Any
    ) -> None:
        """Unlock nodes from *node* up to root."""
        if node is None:
            return

        dec_swa = True
        cur = node
        while cur != self.root_node:
            if cur.lock_ref == 1:
                self._evictable_size += len(cur.key)
                self._protected_size -= len(cur.key)
            cur.lock_ref -= 1

            if self.supports_swa and dec_swa and not cur.swa_tombstone:
                if cur.swa_lock_ref == 1:
                    self._swa_evictable_size += len(cur.key)
                    self._swa_protected_size -= len(cur.key)
                cur.swa_lock_ref -= 1
                if swa_boundary_id and cur.swa_boundary_id == swa_boundary_id:
                    dec_swa = False

            cur = cur.parent

    # ------------------------------------------------------------------
    # Hashing & pretty-print
    # ------------------------------------------------------------------

    def compute_node_hash(self, node: TreeNode) -> List[str]:
        """Compute position-aware SHA-256 hashes for *node* (one per page).

        Lazily computed and cached on ``node.hash_values``.
        """
        if node.hash_values is not None:
            return node.hash_values

        parent_hash: Optional[str] = None
        if (
            node.parent is not None
            and node.parent.hash_values is not None
            and len(node.parent.key) > 0
            and len(node.parent.hash_values) > 0
        ):
            parent_hash = node.parent.hash_values[-1]

        hashes: List[str] = []
        for start in range(0, len(node.key), self.page_size):
            page = node.key.token_ids[start : start + self.page_size]
            if not page:
                continue
            h = hash_token_ids(page, prior_hash=parent_hash)
            hashes.append(h)
            parent_hash = h

        node.hash_values = hashes
        return hashes

    def pretty_print(self) -> None:
        """Print the tree structure to stdout."""
        self._print_helper(self.root_node, 0)
        print(
            f"total={self.total_size()}  evictable={self._evictable_size}"
            + (
                f"  swa_evictable={self._swa_evictable_size}"
                if self.supports_swa
                else ""
            )
        )

    # ------------------------------------------------------------------
    # Internal: match
    # ------------------------------------------------------------------

    def _match_normal(self, key: RadixKey) -> Tuple[List[torch.Tensor], TreeNode]:
        node = self.root_node
        now = time.monotonic()
        node.last_access_time = now
        values: List[torch.Tensor] = []

        while len(key) > 0:
            ck = _child_key(key, self.page_size)
            if ck not in node.children:
                break
            child = node.children[ck]
            child.last_access_time = now
            child.hit_count += 1
            plen = _key_match(child.key, key, self.page_size)
            if plen < len(child.key):
                new_node = self._split_node(child.key, child, plen)
                values.append(new_node.value)
                node = new_node
                break
            values.append(child.value)
            node = child
            key = key[plen:]

        return values, node

    def _match_swa(self, key: RadixKey) -> Tuple[List[torch.Tensor], TreeNode, int]:
        """SWA-aware match.  Returns *(values, last_node, best_value_count)*.

        ``best_value_count`` is the number of value tensors from *values*
        that form a valid SWA-safe prefix (enough non-tombstone tokens within
        the sliding window, or a tombstone-free path to root).
        """
        node = self.root_node
        values: List[torch.Tensor] = []
        non_tomb_len: float = float("inf")
        best_count = 0
        best_node = node

        while len(key) > 0:
            ck = _child_key(key, self.page_size)
            if ck not in node.children:
                break
            child = node.children[ck]

            if child.swa_tombstone:
                if non_tomb_len >= self.sliding_window_size:
                    best_count = len(values)
                    best_node = node
                non_tomb_len = 0

            plen = _key_match(child.key, key, self.page_size)
            if plen < len(child.key):
                new_node = self._split_node(child.key, child, plen)
                values.append(new_node.value)
                if not new_node.swa_tombstone:
                    non_tomb_len += len(new_node.value)
                node = new_node
                break
            values.append(child.value)
            if not child.swa_tombstone:
                non_tomb_len += len(child.value)
            node = child
            key = key[plen:]

        if non_tomb_len >= self.sliding_window_size:
            best_count = len(values)
            best_node = node

        return values, best_node, best_count

    # ------------------------------------------------------------------
    # Internal: insert
    # ------------------------------------------------------------------

    def _insert_normal(
        self, node: TreeNode, key: RadixKey, value: torch.Tensor
    ) -> Tuple[int, TreeNode]:
        """Insert into non-SWA tree.  Returns ``(prefix_len, last_node)``."""
        now = time.monotonic()
        node.last_access_time = now
        if len(key) == 0:
            return 0, node

        total_prefix = 0
        ck = _child_key(key, self.page_size)
        while len(key) > 0 and ck in node.children:
            node = node.children[ck]
            node.last_access_time = now
            plen = _key_match(node.key, key, self.page_size)
            total_prefix += plen
            key = key[plen:]
            value = value[plen:]

            if plen < len(node.key):
                node = self._split_node(node.key, node, plen)
            if len(key) > 0:
                ck = _child_key(key, self.page_size)

        if len(key) > 0:
            new_leaf = self._add_leaf(node, key, value)
            node = new_leaf

        return total_prefix, node

    def _insert_swa(
        self,
        node: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        prev_prefix_len: int,
        swa_evicted_seqlen: int,
    ) -> int:
        """Insert with SWA tombstone awareness.

        When an existing node is tombstoned and the incoming *value* carries
        fresh SWA KV (i.e. beyond *swa_evicted_seqlen*), the node is
        un-tombstoned and its value is replaced.
        """
        now = time.monotonic()
        node.last_access_time = now
        if len(key) == 0:
            return 0

        total_prefix = 0
        while len(key) > 0:
            ck = _child_key(key, self.page_size)
            if ck not in node.children:
                break
            node = node.children[ck]
            node.last_access_time = now
            plen = _key_match(node.key, key, self.page_size)

            if plen < len(node.key):
                self._split_node(node.key, node, plen)

            beyond_protected = prev_prefix_len < total_prefix + plen
            if beyond_protected and node.swa_tombstone:
                if swa_evicted_seqlen <= total_prefix:
                    self._free_indices(node.value[:plen])
                    node.value = value[:plen].clone()
                    node.swa_tombstone = False
                    self._swa_evictable_size += len(node.value)
                else:
                    self._free_indices(value[:plen])
            elif beyond_protected:
                self._free_indices(value[:plen])

            total_prefix += plen
            key = key[plen:]
            value = value[plen:]

        if len(key) > 0:
            if (
                swa_evicted_seqlen > total_prefix
                and swa_evicted_seqlen < total_prefix + len(key)
            ):
                tomb_len = swa_evicted_seqlen - total_prefix
                self._add_leaf(
                    node, key[:tomb_len], value[:tomb_len], swa_tombstone=True
                )
                node = node.children[_child_key(key, self.page_size)]
                key = key[tomb_len:]
                value = value[tomb_len:]

            if len(key) > 0:
                self._add_leaf(node, key, value, swa_tombstone=False)

        return total_prefix

    # ------------------------------------------------------------------
    # Internal: tree manipulation
    # ------------------------------------------------------------------

    def _add_leaf(
        self,
        parent: TreeNode,
        key: RadixKey,
        value: torch.Tensor,
        swa_tombstone: bool = False,
    ) -> TreeNode:
        # Note: we do NOT subtract parent's tokens from _evictable_size
        # when a leaf gains its first child, even though the parent is no
        # longer directly evictable as a leaf.  Reasons:
        #
        # 1. Internal nodes ARE reclaimable via cascade eviction: evict()
        #    evicts all children first, then the childless parent cascades.
        #    So _evictable_size correctly tracks "total reclaimable tokens".
        #
        # 2. _split_node (which also creates internal nodes) does not adjust
        #    _evictable_size.  Subtracting here but not there would create
        #    an inconsistency.
        #
        # 3. inc_lock_ref / dec_lock_ref assume ALL non-root tokens are
        #    partitioned into evictable + protected.  Subtracting here
        #    breaks that invariant, causing _evictable_size to go negative
        #    when locks walk up through internal nodes whose tokens were
        #    already removed.
        #
        # A safety guard in alloc_kv_with_eviction() breaks the eviction
        # loop if evict() frees 0 tokens despite evictable_size > 0.

        new_node = TreeNode()
        new_node.parent = parent
        new_node.key = key
        new_node.value = value.clone()
        new_node.swa_tombstone = swa_tombstone
        parent.children[_child_key(key, self.page_size)] = new_node
        self._evictable_size += len(key)
        if self.supports_swa and not swa_tombstone:
            self._swa_evictable_size += len(key)
        return new_node

    def _split_node(self, key: RadixKey, child: TreeNode, split_len: int) -> TreeNode:
        """Split *child* at *split_len*, returning the new parent node."""
        logger.debug(
            "[SPLIT] node_id=%d key_len=%d split_len=%d "
            "parent_val[:4]=%s child_val[:4]=%s",
            child.id,
            len(key),
            split_len,
            child.value[:min(split_len, 4)].tolist()
            if child.value is not None
            else [],
            child.value[split_len : split_len + 4].tolist()
            if child.value is not None and len(child.value) > split_len
            else [],
        )
        new_node = TreeNode()
        new_node.children[_child_key(key[split_len:], self.page_size)] = child
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.swa_lock_ref = child.swa_lock_ref
        new_node.swa_tombstone = child.swa_tombstone
        new_node.swa_boundary_id = child.swa_boundary_id
        child.swa_boundary_id = None
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len].clone()

        # Split hash values if they exist
        if child.hash_values is not None:
            pages = split_len // self.page_size if self.page_size > 1 else split_len
            new_node.hash_values = child.hash_values[:pages]
            child.hash_values = child.hash_values[pages:]
        else:
            new_node.hash_values = None

        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:].clone()
        new_node.parent.children[_child_key(key, self.page_size)] = new_node
        return new_node

    def _delete_leaf(self, node: TreeNode) -> None:
        ck = _child_key(node.key, self.page_size)
        node.parent.children.pop(ck, None)
        self._evictable_size -= len(node.key)
        if self.supports_swa and not node.swa_tombstone:
            self._swa_evictable_size -= len(node.key)
        node.value = None
        if self.on_node_evict is not None:
            self.on_node_evict(node.id)

    def _tombstone_node(self, node: TreeNode) -> None:
        node.swa_tombstone = True
        self._swa_evictable_size -= len(node.key)

    # ------------------------------------------------------------------
    # Internal: collection helpers
    # ------------------------------------------------------------------

    def _collect_evictable_leaves(self) -> List[TreeNode]:
        leaves: List[TreeNode] = []
        stack: List[TreeNode] = [self.root_node]
        while stack:
            n = stack.pop()
            if n.evicted:
                continue
            has_live_child = False
            for c in n.children.values():
                if not c.evicted:
                    has_live_child = True
                    stack.append(c)
            if not has_live_child and n.lock_ref == 0 and n != self.root_node:
                leaves.append(n)
        return leaves

    def _collect_swa_evictable(self) -> List[TreeNode]:
        nodes: List[TreeNode] = []
        stack: List[TreeNode] = [self.root_node]
        while stack:
            n = stack.pop()
            if n.evicted:
                continue
            if n != self.root_node and not n.swa_tombstone and n.swa_lock_ref == 0:
                nodes.append(n)
            stack.extend(c for c in n.children.values() if not c.evicted)
        return nodes

    def _page_align_key(self, key: RadixKey) -> RadixKey:
        if self.page_size == 1:
            return key
        aligned = len(key) // self.page_size * self.page_size
        return key[:aligned]

    def _free_indices(self, indices: torch.Tensor) -> None:
        if self.pool is not None and len(indices) > 0:
            self.pool.free(indices)

    def _free_swa_indices(self, indices: torch.Tensor) -> None:
        if self.pool is not None and len(indices) > 0:
            if hasattr(self.pool, "free_swa"):
                self.pool.free_swa(indices)
            else:
                self.pool.free(indices)

    def _print_helper(self, node: TreeNode, indent: int) -> None:
        stack = [(node, indent)]
        while stack:
            n, ind = stack.pop()
            toks = n.key.token_ids[:10] if n.key else []
            klen = len(n.key) if n.key else 0
            flags = f"lock={n.lock_ref}"
            if self.supports_swa:
                flags += f" swa={n.swa_lock_ref} tomb={n.swa_tombstone}"
            print(f"{'  ' * ind}[{klen}] {toks} {flags}")
            for c in n.children.values():
                stack.append((c, ind + 1))
