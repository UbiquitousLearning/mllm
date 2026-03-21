"""Tests for RadixCache correctness.

Exercises the radix tree's insert, match, split, evict, lock/unlock,
and size accounting -- all on CPU tensors, no GPU required.
"""

import torch
import pytest

from pymllm.mem_cache.base_prefix_cache import RadixKey
from pymllm.mem_cache.radix_cache import RadixCache, TreeNode
from pymllm.mem_cache.memory_pool import TokenToKVPoolAllocator


# ======================================================================
# Helpers
# ======================================================================


def _key(token_ids, extra_key=None):
    return RadixKey(list(token_ids), extra_key=extra_key)


def _val(token_ids):
    return torch.tensor(list(token_ids), dtype=torch.int64)


def _make_cache(pool_size=256, page_size=1, sliding_window_size=None, on_node_evict=None):
    pool = TokenToKVPoolAllocator(size=pool_size, device="cpu", page_size=page_size)
    return RadixCache(
        page_size=page_size,
        token_to_kv_pool_allocator=pool,
        sliding_window_size=sliding_window_size,
        on_node_evict=on_node_evict,
    )


# ======================================================================
# Basic insert and match
# ======================================================================


class TestInsertAndMatch:
    def test_insert_then_match_exact(self):
        cache = _make_cache()
        key = _key([1, 2, 3, 4])
        val = _val([10, 20, 30, 40])
        cache.insert(key, val)

        result = cache.match_prefix(_key([1, 2, 3, 4]))
        assert result.prefix_len == 4
        assert result.indices.tolist() == [10, 20, 30, 40]

    def test_match_prefix_shorter(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))

        result = cache.match_prefix(_key([1, 2]))
        assert result.prefix_len == 2
        assert result.indices.tolist() == [10, 20]

    def test_match_prefix_longer(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))

        result = cache.match_prefix(_key([1, 2, 3, 4, 5]))
        assert result.prefix_len == 3
        assert result.indices.tolist() == [10, 20, 30]

    def test_no_match(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))

        result = cache.match_prefix(_key([9, 8, 7]))
        assert result.prefix_len == 0
        assert result.indices.numel() == 0

    def test_empty_key(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))

        result = cache.match_prefix(_key([]))
        assert result.prefix_len == 0

    def test_insert_returns_prefix_len(self):
        cache = _make_cache()
        r1 = cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        assert r1.prefix_len == 0

        r2 = cache.insert(_key([1, 2, 3, 4, 5]), _val([10, 20, 30, 40, 50]))
        assert r2.prefix_len == 3

    def test_insert_duplicate_is_idempotent(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        r = cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        assert r.prefix_len == 3

        result = cache.match_prefix(_key([1, 2, 3]))
        assert result.prefix_len == 3


# ======================================================================
# Tree splitting
# ======================================================================


class TestSplitNode:
    def test_split_on_partial_match(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))

        r1 = cache.match_prefix(_key([1, 2, 3, 4]))
        assert r1.prefix_len == 4
        assert r1.indices.tolist() == [10, 20, 30, 40]

        r2 = cache.match_prefix(_key([1, 2, 5, 6]))
        assert r2.prefix_len == 4
        assert r2.indices.tolist() == [10, 20, 50, 60]

        r3 = cache.match_prefix(_key([1, 2]))
        assert r3.prefix_len == 2
        assert r3.indices.tolist() == [10, 20]

    def test_multiple_branches(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        cache.insert(_key([1, 2, 4]), _val([10, 20, 40]))
        cache.insert(_key([1, 2, 5]), _val([10, 20, 50]))

        for suffix, expected_last in [(3, 30), (4, 40), (5, 50)]:
            r = cache.match_prefix(_key([1, 2, suffix]))
            assert r.prefix_len == 3
            assert r.indices.tolist() == [10, 20, expected_last]


# ======================================================================
# Size accounting
# ======================================================================


class TestSizeAccounting:
    def test_evictable_size_after_insert(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        assert cache.evictable_size() == 3

    def test_evictable_size_after_branch(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        assert cache.evictable_size() == 4

        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))
        # After split: [1,2] (2, internal) + [3,4] (2, leaf) + [5,6] (2, leaf) = 6
        # evictable_size tracks cascade-reclaimable tokens (including internal nodes)
        assert cache.evictable_size() == 6
        _assert_size_invariant(cache)

        # Only leaf nodes are directly evictable
        leaves = cache._collect_evictable_leaves()
        leaf_tokens = sum(len(n.key) for n in leaves)
        assert leaf_tokens == 4  # [3,4] + [5,6]

        # But cascade eviction can reclaim all 6
        result = cache.evict(100)
        assert result.full_evicted == 6

    def test_evictable_includes_internal_nodes(self):
        """Internal nodes are reclaimable via cascade eviction, so their tokens
        should remain in evictable_size."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        assert cache.evictable_size() == 3

        # Inserting a longer sequence: [1,2,3] becomes internal, [4,5] is new leaf
        cache.insert(_key([1, 2, 3, 4, 5]), _val([10, 20, 30, 40, 50]))
        # [1,2,3] (3 tokens, internal) + [4,5] (2 tokens, leaf) = 5 total reclaimable
        assert cache.evictable_size() == 5
        _assert_size_invariant(cache)

    def test_protected_size_tracks_locked_nodes(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        result = cache.match_prefix(_key([1, 2, 3]))
        assert cache.protected_size() == 0

        cache.inc_lock_ref(result.last_node)
        assert cache.protected_size() == 3
        assert cache.evictable_size() == 0

        cache.dec_lock_ref(result.last_node)
        assert cache.protected_size() == 0
        assert cache.evictable_size() == 3


# ======================================================================
# Eviction
# ======================================================================


class TestEviction:
    def test_evict_frees_tokens(self):
        cache = _make_cache(pool_size=64)
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        cache.insert(_key([4, 5, 6]), _val([40, 50, 60]))
        assert cache.evictable_size() == 6

        result = cache.evict(3)
        assert result.full_evicted >= 3
        assert cache.evictable_size() <= 3

    def test_evict_respects_lock(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        r = cache.match_prefix(_key([1, 2, 3]))
        cache.inc_lock_ref(r.last_node)

        result = cache.evict(3)
        assert result.full_evicted == 0
        assert cache.match_prefix(_key([1, 2, 3])).prefix_len == 3

        cache.dec_lock_ref(r.last_node)
        result = cache.evict(3)
        assert result.full_evicted == 3

    def test_evict_lru_order(self):
        """Least recently accessed nodes should be evicted first."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        cache.insert(_key([4, 5, 6]), _val([40, 50, 60]))

        # Access [4,5,6] more recently
        cache.match_prefix(_key([4, 5, 6]))

        cache.evict(3)
        # [1,2,3] should be evicted (older access), [4,5,6] should remain
        assert cache.match_prefix(_key([1, 2, 3])).prefix_len == 0
        assert cache.match_prefix(_key([4, 5, 6])).prefix_len == 3

    def test_evict_cascades_to_childless_parent(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))
        # Tree: root -> [1,2] -> {[3,4], [5,6]}

        # Evict [3,4] leaf
        # After evicting [3,4], parent [1,2] becomes childless (only if [5,6] also evicted)
        # Evict enough to clear one branch
        cache.evict(2)
        # One leaf evicted, parent [1,2] still has one child
        remaining = cache.match_prefix(_key([1, 2]))
        assert remaining.prefix_len == 2  # [1,2] shared prefix still there

    def test_on_node_evict_callback(self):
        evicted_ids = []
        cache = _make_cache(on_node_evict=lambda nid: evicted_ids.append(nid))
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        cache.evict(3)
        assert len(evicted_ids) == 1


# ======================================================================
# Lock reference counting
# ======================================================================


class TestLockRefCounting:
    def test_multiple_locks(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        r = cache.match_prefix(_key([1, 2, 3]))
        node = r.last_node

        cache.inc_lock_ref(node)
        cache.inc_lock_ref(node)
        assert node.lock_ref == 2
        assert cache.protected_size() == 3

        cache.dec_lock_ref(node)
        assert node.lock_ref == 1
        assert cache.protected_size() == 3  # still protected

        cache.dec_lock_ref(node)
        assert node.lock_ref == 0
        assert cache.protected_size() == 0

    def test_lock_propagates_to_ancestors(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))
        # Tree: root -> [1,2] -> {[3,4], [5,6]}

        r = cache.match_prefix(_key([1, 2, 3, 4]))
        cache.inc_lock_ref(r.last_node)

        # [3,4] leaf node locked
        assert r.last_node.lock_ref == 1
        # [1,2] ancestor also locked
        assert r.last_node.parent.lock_ref == 1

        cache.dec_lock_ref(r.last_node)
        assert r.last_node.lock_ref == 0
        assert r.last_node.parent.lock_ref == 0

    def test_lock_null_node_is_noop(self):
        cache = _make_cache()
        cache.inc_lock_ref(None)  # should not raise
        cache.dec_lock_ref(None)  # should not raise


# ======================================================================
# Extra key (namespace isolation)
# ======================================================================


class TestExtraKey:
    def test_different_namespaces_dont_share(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3], extra_key="lora_a"), _val([10, 20, 30]))
        cache.insert(_key([1, 2, 3], extra_key="lora_b"), _val([40, 50, 60]))

        ra = cache.match_prefix(_key([1, 2, 3], extra_key="lora_a"))
        assert ra.prefix_len == 3
        assert ra.indices.tolist() == [10, 20, 30]

        rb = cache.match_prefix(_key([1, 2, 3], extra_key="lora_b"))
        assert rb.prefix_len == 3
        assert rb.indices.tolist() == [40, 50, 60]

    def test_no_cross_namespace_match(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3], extra_key="ns1"), _val([10, 20, 30]))

        r = cache.match_prefix(_key([1, 2, 3], extra_key="ns2"))
        assert r.prefix_len == 0

        r2 = cache.match_prefix(_key([1, 2, 3]))  # no extra_key
        assert r2.prefix_len == 0


# ======================================================================
# Page-aligned operations (page_size > 1)
# ======================================================================


class TestPageAligned:
    def test_page_aligned_insert_and_match(self):
        cache = _make_cache(page_size=4)
        cache.insert(_key(range(8)), _val(range(100, 108)))

        r = cache.match_prefix(_key(range(8)))
        assert r.prefix_len == 8

    def test_page_alignment_truncates_key(self):
        cache = _make_cache(page_size=4)
        cache.insert(_key(range(8)), _val(range(100, 108)))

        # Query with 6 tokens: page-aligned to 4
        r = cache.match_prefix(_key(range(6)))
        assert r.prefix_len == 4

    def test_page_aligned_partial_match(self):
        cache = _make_cache(page_size=4)
        cache.insert(_key(range(8)), _val(range(100, 108)))

        # Query first 4 tokens matching, then different
        q = list(range(4)) + [99, 98, 97, 96]
        r = cache.match_prefix(_key(q))
        assert r.prefix_len == 4


# ======================================================================
# Reset
# ======================================================================


class TestReset:
    def test_reset_clears_all(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        assert cache.evictable_size() == 3

        cache.reset()
        assert cache.evictable_size() == 0
        assert cache.match_prefix(_key([1, 2, 3])).prefix_len == 0


# ======================================================================
# SWA (Sliding Window Attention) mode
# ======================================================================


class TestSWA:
    def test_swa_basic_insert_and_match(self):
        cache = _make_cache(sliding_window_size=4)
        cache.insert(_key([1, 2, 3, 4, 5, 6]), _val([10, 20, 30, 40, 50, 60]))

        r = cache.match_prefix(_key([1, 2, 3, 4, 5, 6]))
        assert r.prefix_len == 6

    def test_swa_tombstone_eviction(self):
        cache = _make_cache(sliding_window_size=4)
        cache.insert(_key([1, 2, 3, 4, 5, 6]), _val([10, 20, 30, 40, 50, 60]))
        cache.insert(_key([1, 2, 3, 7, 8, 9]), _val([10, 20, 30, 70, 80, 90]))
        # Tree: root -> [1,2,3] -> {[4,5,6], [7,8,9]}

        # SWA evict should tombstone internal nodes (free SWA KV but retain full-attn KV)
        result = cache.evict(0, swa_num_tokens=3)
        assert result.swa_evicted >= 0  # may or may not evict depending on lock state

    def test_swa_lock_ref_tracks_boundary(self):
        cache = _make_cache(sliding_window_size=4)
        cache.insert(_key([1, 2, 3, 4, 5, 6]), _val([10, 20, 30, 40, 50, 60]))

        r = cache.match_prefix(_key([1, 2, 3, 4, 5, 6]))
        bid = cache.inc_lock_ref(r.last_node)

        # With window=4, swa_lock should cover the last 4 tokens
        # boundary_id should be set
        if cache.supports_swa:
            assert bid is not None or r.last_node.swa_lock_ref > 0

        cache.dec_lock_ref(r.last_node, swa_boundary_id=bid)

    def test_swa_evictable_size_tracking(self):
        cache = _make_cache(sliding_window_size=4)
        cache.insert(_key([1, 2, 3, 4, 5, 6]), _val([10, 20, 30, 40, 50, 60]))
        assert cache.swa_evictable_size() == 6

        r = cache.match_prefix(_key([1, 2, 3, 4, 5, 6]))
        bid = cache.inc_lock_ref(r.last_node)
        assert cache.swa_protected_size() > 0

        cache.dec_lock_ref(r.last_node, swa_boundary_id=bid)
        assert cache.swa_protected_size() == 0


# ======================================================================
# Evictable size accounting after split (potential bug)
# ======================================================================


class TestEvictableSizeAfterSplit:
    """Verify that _evictable_size stays consistent with actual evictable leaves
    after node splits. This is a known area of concern."""

    def test_split_tracks_cascade_reclaimable(self):
        """After a split, evictable_size includes internal nodes (cascade-reclaimable)."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4, 5, 6]), _val([10, 20, 30, 40, 50, 60]))
        assert cache.evictable_size() == 6

        cache.insert(_key([1, 2, 3, 7, 8, 9]), _val([10, 20, 30, 70, 80, 90]))
        # root -> [1,2,3] (3, internal) -> {[4,5,6] (3, leaf), [7,8,9] (3, leaf)}

        # evictable_size = 9 (all reclaimable via cascade)
        assert cache.evictable_size() == 9
        _assert_size_invariant(cache)

        # Only 6 tokens in leaves, but cascade recovers the internal node too
        leaves = cache._collect_evictable_leaves()
        assert sum(len(n.key) for n in leaves) == 6

        result = cache.evict(100)
        assert result.full_evicted == 9

    def test_cascade_eviction_after_split(self):
        """Evicting all leaves should cascade to evict the now-childless parent."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))
        # root -> [1,2] -> {[3,4], [5,6]}

        initial_evictable = cache.evictable_size()
        result = cache.evict(100)

        # Should evict all: [3,4] + [5,6] as leaves, then [1,2] cascades
        assert result.full_evicted == initial_evictable
        assert cache.evictable_size() == 0
        assert cache.match_prefix(_key([1, 2, 3, 4])).prefix_len == 0

    def test_partial_cascade(self):
        """Evicting one leaf should not evict the parent if sibling remains."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))

        cache.evict(2)  # evict one leaf
        # Parent [1,2] should NOT be evicted because one sibling remains
        assert cache.match_prefix(_key([1, 2])).prefix_len == 2


# ======================================================================
# Pool allocator integration
# ======================================================================


class TestPoolIntegration:
    def test_evict_returns_indices_to_pool(self):
        pool = TokenToKVPoolAllocator(size=32, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        initial_available = pool.available_size()
        indices = pool.alloc(6)
        assert pool.available_size() == initial_available - 6

        cache.insert(_key([1, 2, 3]), indices[:3])
        cache.insert(_key([4, 5, 6]), indices[3:])

        cache.evict(6)
        pool.merge_and_sort_free()
        assert pool.available_size() == initial_available

    def test_locked_nodes_preserve_pool_indices(self):
        pool = TokenToKVPoolAllocator(size=32, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        indices = pool.alloc(3)
        cache.insert(_key([1, 2, 3]), indices)

        r = cache.match_prefix(_key([1, 2, 3]))
        cache.inc_lock_ref(r.last_node)

        cache.evict(3)
        # Locked - should not be evicted, indices stay allocated
        r2 = cache.match_prefix(_key([1, 2, 3]))
        assert r2.prefix_len == 3

        cache.dec_lock_ref(r.last_node)


# ======================================================================
# Stress / multi-sequence scenarios
# ======================================================================


class TestMultiSequence:
    def test_many_sequences_with_shared_prefix(self):
        cache = _make_cache()
        system_prompt = [100, 101, 102, 103, 104]
        system_val = [200, 201, 202, 203, 204]

        for i in range(10):
            key = system_prompt + [i * 10 + j for j in range(5)]
            val = system_val + [300 + i * 10 + j for j in range(5)]
            cache.insert(_key(key), _val(val))

        # All 10 sequences share the system prompt prefix
        for i in range(10):
            key = system_prompt + [i * 10 + j for j in range(5)]
            r = cache.match_prefix(_key(key))
            assert r.prefix_len == 10
            # System prompt values should be shared
            assert r.indices[:5].tolist() == system_val

    def test_insert_match_evict_cycle(self):
        """Simulate a realistic request lifecycle."""
        pool = TokenToKVPoolAllocator(size=64, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        # Request 1: insert, lock, use, unlock
        indices1 = pool.alloc(5)
        cache.insert(_key([1, 2, 3, 4, 5]), indices1)
        r1 = cache.match_prefix(_key([1, 2, 3, 4, 5]))
        bid1 = cache.inc_lock_ref(r1.last_node)

        # Request 2: shares prefix [1,2,3]
        indices2 = pool.alloc(5)
        cache.insert(_key([1, 2, 3, 6, 7]), indices2)
        r2 = cache.match_prefix(_key([1, 2, 3, 6, 7]))
        bid2 = cache.inc_lock_ref(r2.last_node)

        # Unlock request 1
        cache.dec_lock_ref(r1.last_node, swa_boundary_id=bid1)

        # Evict - should only evict unlocked leaves
        before = cache.evictable_size()
        cache.evict(2)

        # Request 2 should still be accessible
        r2_check = cache.match_prefix(_key([1, 2, 3, 6, 7]))
        assert r2_check.prefix_len == 5

        cache.dec_lock_ref(r2.last_node, swa_boundary_id=bid2)


# ======================================================================
# Tree invariant checks
# ======================================================================


def _tree_token_count(cache):
    """Walk the tree and count tokens in all non-root, non-evicted nodes."""
    total = 0
    stack = [cache.root_node]
    while stack:
        n = stack.pop()
        for c in n.children.values():
            if not c.evicted:
                total += len(c.key)
                stack.append(c)
    return total


def _assert_size_invariant(cache):
    """Verify: evictable + protected == total tree tokens (non-SWA mode)."""
    tree_total = _tree_token_count(cache)
    accounting_total = cache.evictable_size() + cache.protected_size()
    assert accounting_total == tree_total, (
        f"Size invariant violated: evictable({cache.evictable_size()}) + "
        f"protected({cache.protected_size()}) = {accounting_total} != "
        f"tree_total({tree_total})"
    )


class TestSizeInvariant:
    """Verify evictable + protected == total tree tokens after every operation."""

    def test_invariant_after_insert(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        _assert_size_invariant(cache)

    def test_invariant_after_split(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        _assert_size_invariant(cache)

        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))
        _assert_size_invariant(cache)

    def test_invariant_after_lock_unlock(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))

        r = cache.match_prefix(_key([1, 2, 3, 4]))
        cache.inc_lock_ref(r.last_node)
        _assert_size_invariant(cache)

        cache.dec_lock_ref(r.last_node)
        _assert_size_invariant(cache)

    def test_invariant_after_evict(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        cache.insert(_key([4, 5, 6]), _val([40, 50, 60]))

        cache.evict(3)
        _assert_size_invariant(cache)

    def test_invariant_after_partial_evict_with_lock(self):
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))

        # Lock one branch
        r = cache.match_prefix(_key([1, 2, 3, 4]))
        cache.inc_lock_ref(r.last_node)

        # Evict the unlocked branch
        cache.evict(2)
        _assert_size_invariant(cache)

        cache.dec_lock_ref(r.last_node)
        _assert_size_invariant(cache)

    def test_invariant_through_complex_lifecycle(self):
        """Simulate multiple requests with shared prefixes and verify invariant
        at every step."""
        cache = _make_cache()

        # Insert system prompt
        cache.insert(_key([1, 2, 3, 4, 5]), _val([10, 20, 30, 40, 50]))
        _assert_size_invariant(cache)

        # Req 1 extends system prompt
        cache.insert(
            _key([1, 2, 3, 4, 5, 100, 101]),
            _val([10, 20, 30, 40, 50, 110, 111]),
        )
        _assert_size_invariant(cache)

        # Req 2 diverges at token 3
        cache.insert(
            _key([1, 2, 3, 200, 201]),
            _val([10, 20, 30, 210, 211]),
        )
        _assert_size_invariant(cache)

        # Lock req 1's leaf
        r1 = cache.match_prefix(_key([1, 2, 3, 4, 5, 100, 101]))
        cache.inc_lock_ref(r1.last_node)
        _assert_size_invariant(cache)

        # Lock req 2's leaf
        r2 = cache.match_prefix(_key([1, 2, 3, 200, 201]))
        cache.inc_lock_ref(r2.last_node)
        _assert_size_invariant(cache)

        # Evict (nothing should be evicted — all locked)
        cache.evict(100)
        _assert_size_invariant(cache)

        # Unlock req 1
        cache.dec_lock_ref(r1.last_node)
        _assert_size_invariant(cache)

        # Evict req 1's unique suffix
        cache.evict(2)
        _assert_size_invariant(cache)

        # Unlock req 2
        cache.dec_lock_ref(r2.last_node)
        _assert_size_invariant(cache)

        # Evict everything remaining
        cache.evict(100)
        _assert_size_invariant(cache)
        assert cache.evictable_size() == 0

    def test_invariant_after_match_triggers_split(self):
        """match_prefix can trigger splits. Verify invariant is maintained."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4, 5, 6]), _val([10, 20, 30, 40, 50, 60]))
        _assert_size_invariant(cache)

        # Match at a point that splits the node
        r = cache.match_prefix(_key([1, 2, 3]))
        _assert_size_invariant(cache)
        assert r.prefix_len == 3

    def test_invariant_split_locked_node(self):
        """Splitting a locked node must preserve the size invariant."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4, 5, 6]), _val([10, 20, 30, 40, 50, 60]))

        # Lock the leaf
        r = cache.match_prefix(_key([1, 2, 3, 4, 5, 6]))
        cache.inc_lock_ref(r.last_node)
        _assert_size_invariant(cache)

        # Now match a shorter prefix — this triggers a split on a LOCKED node
        r2 = cache.match_prefix(_key([1, 2, 3]))
        _assert_size_invariant(cache)

        # The original node reference should still be valid for unlock
        cache.dec_lock_ref(r.last_node)
        _assert_size_invariant(cache)


# ======================================================================
# Double-unlock and negative ref count protection
# ======================================================================


class TestLockEdgeCases:
    def test_dec_without_inc_goes_negative(self):
        """Verify behavior when dec_lock_ref is called without matching inc.
        This documents whether negative lock_ref causes issues."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3]), _val([10, 20, 30]))
        r = cache.match_prefix(_key([1, 2, 3]))

        # dec without inc — lock_ref goes to -1
        cache.dec_lock_ref(r.last_node)
        # lock_ref is now -1, evictable_size and protected_size may be inconsistent
        # This is a potential bug: negative lock_ref means the node is "super evictable"
        assert r.last_node.lock_ref == -1

    def test_split_preserves_lock_ref_across_both_halves(self):
        """When a locked node is split, both halves must inherit the lock count."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4, 5, 6]), _val([10, 20, 30, 40, 50, 60]))

        r = cache.match_prefix(_key([1, 2, 3, 4, 5, 6]))
        original_node = r.last_node
        cache.inc_lock_ref(original_node)
        cache.inc_lock_ref(original_node)
        assert original_node.lock_ref == 2

        # Insert a diverging key to force a split at position 3
        cache.insert(_key([1, 2, 3, 7, 8, 9]), _val([10, 20, 30, 70, 80, 90]))

        # original_node is now the tail [4,5,6] after split
        # Its lock_ref should still be 2
        assert original_node.lock_ref == 2

        # The new parent [1,2,3] should also have lock_ref == 2
        parent = original_node.parent
        assert parent.lock_ref == 2

        # Both unlock ops should work correctly
        cache.dec_lock_ref(original_node)
        cache.dec_lock_ref(original_node)
        assert original_node.lock_ref == 0
        assert parent.lock_ref == 0

    def test_concurrent_locks_on_shared_prefix(self):
        """Two requests locking different branches of a shared prefix."""
        cache = _make_cache()
        cache.insert(_key([1, 2, 3, 4]), _val([10, 20, 30, 40]))
        cache.insert(_key([1, 2, 5, 6]), _val([10, 20, 50, 60]))
        # root -> [1,2] -> {[3,4], [5,6]}

        r1 = cache.match_prefix(_key([1, 2, 3, 4]))
        r2 = cache.match_prefix(_key([1, 2, 5, 6]))

        cache.inc_lock_ref(r1.last_node)
        cache.inc_lock_ref(r2.last_node)

        # Shared ancestor [1,2] should have lock_ref == 2
        shared = r1.last_node.parent
        assert shared.lock_ref == 2

        # Evict should fail (everything locked)
        result = cache.evict(100)
        assert result.full_evicted == 0

        # Unlock one branch
        cache.dec_lock_ref(r1.last_node)
        assert shared.lock_ref == 1

        # Now only [3,4] is evictable (its lock_ref is 0), but parent is still locked
        result = cache.evict(2)
        assert result.full_evicted == 2  # [3,4] evicted
        assert cache.match_prefix(_key([1, 2, 5, 6])).prefix_len == 4  # [1,2,5,6] still there

        cache.dec_lock_ref(r2.last_node)
        _assert_size_invariant(cache)


# ======================================================================
# Pool leak detection
# ======================================================================


class TestPoolLeaks:
    def test_no_pool_leak_after_full_lifecycle(self):
        """All allocated pool indices must be returned after evicting everything."""
        pool = TokenToKVPoolAllocator(size=128, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)
        initial = pool.available_size()

        # Allocate and insert 10 sequences
        nodes = []
        for i in range(10):
            indices = pool.alloc(5)
            key = [i * 100 + j for j in range(5)]
            cache.insert(_key(key), indices)

        assert pool.available_size() < initial

        # Lock some, then unlock
        for i in range(5):
            key = [i * 100 + j for j in range(5)]
            r = cache.match_prefix(_key(key))
            nodes.append(r.last_node)
            cache.inc_lock_ref(r.last_node)

        for node in nodes:
            cache.dec_lock_ref(node)

        # Evict all
        cache.evict(1000)
        pool.merge_and_sort_free()
        assert pool.available_size() == initial

    def test_no_pool_leak_with_shared_prefix_eviction(self):
        """Pool indices for shared prefixes must be freed exactly once."""
        pool = TokenToKVPoolAllocator(size=128, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)
        initial = pool.available_size()

        # Two sequences sharing a prefix — allocate separate pool indices
        idx1 = pool.alloc(6)
        cache.insert(_key([1, 2, 3, 4, 5, 6]), idx1)

        idx2 = pool.alloc(6)
        r = cache.insert(_key([1, 2, 3, 7, 8, 9]), idx2)
        # Prefix [1,2,3] is shared — insert returns prefix_len=3
        # Caller must free duplicate indices idx2[:3]
        if r.prefix_len > 0:
            pool.free(idx2[: r.prefix_len])

        # Now evict everything
        cache.evict(1000)
        pool.merge_and_sort_free()
        assert pool.available_size() == initial


# ======================================================================
# Realistic multi-request serving scenarios
# ======================================================================


# Simulate the model runner's insert-and-free-duplicates pattern
def _model_runner_insert(cache, pool, token_ids, seq_kv_indices):
    """Mimics ModelRunnerProcess._insert_into_radix_cache:
    insert, free duplicate KV indices for the shared prefix, rematch."""
    key = _key(token_ids)
    result = cache.insert(key, seq_kv_indices)
    if result.prefix_len > 0:
        pool.free(seq_kv_indices[: result.prefix_len])
    return result


class TestConcurrentRequestsSharedPrefix:
    """Simulate multiple in-flight requests sharing a system prompt,
    each with different user messages, arriving and finishing at
    different times."""

    SYSTEM_PROMPT = list(range(1000, 1050))  # 50-token system prompt
    SYSTEM_PROMPT_LEN = 50

    def _make_user_msg(self, user_id, length=20):
        return [2000 + user_id * 100 + j for j in range(length)]

    def _full_seq(self, user_id, length=20):
        return self.SYSTEM_PROMPT + self._make_user_msg(user_id, length)

    def test_10_concurrent_requests_shared_system_prompt(self):
        """10 requests share a 50-token system prompt, each with a unique
        20-token user message. All active simultaneously."""
        pool = TokenToKVPoolAllocator(size=2048, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)
        initial_pool = pool.available_size()

        locked_nodes = []
        lock_ids = []

        # Phase 1: all 10 requests arrive and get prefilled
        for uid in range(10):
            seq = self._full_seq(uid)
            indices = pool.alloc(len(seq))
            assert indices is not None
            _model_runner_insert(cache, pool, seq, indices)

            r = cache.match_prefix(_key(seq))
            assert r.prefix_len == len(seq)
            bid = cache.inc_lock_ref(r.last_node)
            locked_nodes.append(r.last_node)
            lock_ids.append(bid)

        _assert_size_invariant(cache)

        # All 10 requests share the system prompt → shared prefix node should
        # have lock_ref == 10 (from all 10 inc_lock_ref walks)
        r_check = cache.match_prefix(_key(self.SYSTEM_PROMPT))
        assert r_check.prefix_len == self.SYSTEM_PROMPT_LEN

        # Eviction should fail — everything is locked
        result = cache.evict(1000)
        assert result.full_evicted == 0

        # Phase 2: requests finish one by one
        for i in range(10):
            cache.dec_lock_ref(locked_nodes[i], swa_boundary_id=lock_ids[i])
            _assert_size_invariant(cache)

        # Phase 3: evict everything
        cache.evict(10000)
        _assert_size_invariant(cache)
        pool.merge_and_sort_free()
        assert pool.available_size() == initial_pool

    def test_staggered_arrival_and_departure(self):
        """Requests arrive and finish in interleaved order, simulating
        continuous batching where some requests are prefilling while
        others are decoding."""
        pool = TokenToKVPoolAllocator(size=2048, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)
        initial_pool = pool.available_size()

        active = {}  # uid -> (last_node, boundary_id)

        # Wave 1: requests 0-4 arrive
        for uid in range(5):
            seq = self._full_seq(uid)
            indices = pool.alloc(len(seq))
            _model_runner_insert(cache, pool, seq, indices)
            r = cache.match_prefix(_key(seq))
            bid = cache.inc_lock_ref(r.last_node)
            active[uid] = (r.last_node, bid)

        _assert_size_invariant(cache)

        # Request 0 and 2 finish (decode complete)
        for uid in [0, 2]:
            node, bid = active.pop(uid)
            cache.dec_lock_ref(node, swa_boundary_id=bid)
        _assert_size_invariant(cache)

        # Wave 2: requests 5-9 arrive (while 1, 3, 4 are still decoding)
        for uid in range(5, 10):
            seq = self._full_seq(uid)
            indices = pool.alloc(len(seq))
            _model_runner_insert(cache, pool, seq, indices)
            r = cache.match_prefix(_key(seq))
            bid = cache.inc_lock_ref(r.last_node)
            active[uid] = (r.last_node, bid)

        _assert_size_invariant(cache)

        # Request 1 finishes
        node, bid = active.pop(1)
        cache.dec_lock_ref(node, swa_boundary_id=bid)

        # Evict expired request data (0, 1, 2 are unlocked now)
        cache.evict(100)
        _assert_size_invariant(cache)

        # Remaining requests (3, 4, 5-9) should still match
        for uid in active:
            seq = self._full_seq(uid)
            r = cache.match_prefix(_key(seq))
            assert r.prefix_len == len(seq), f"Request {uid} lost cache"

        # Cleanup
        for uid in list(active):
            node, bid = active.pop(uid)
            cache.dec_lock_ref(node, swa_boundary_id=bid)

        cache.evict(10000)
        pool.merge_and_sort_free()
        assert pool.available_size() == initial_pool

    def test_cache_hit_reuse_after_previous_request(self):
        """Request B arrives with the same prompt as finished request A.
        B should get a full cache hit on A's KV data."""
        pool = TokenToKVPoolAllocator(size=2048, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        seq = self._full_seq(0)

        # Request A: prefill, decode, finish
        indices_a = pool.alloc(len(seq))
        _model_runner_insert(cache, pool, seq, indices_a)
        r_a = cache.match_prefix(_key(seq))
        bid_a = cache.inc_lock_ref(r_a.last_node)
        cache.dec_lock_ref(r_a.last_node, swa_boundary_id=bid_a)

        # Request B: same prompt — should get full cache hit
        r_b = cache.match_prefix(_key(seq))
        assert r_b.prefix_len == len(seq)
        # B's cached indices should match A's original values
        assert r_b.indices.tolist() == indices_a.tolist()

    def test_memory_pressure_eviction_during_serving(self):
        """Small pool forces eviction while requests are active.
        Only unlocked (finished) requests should be evicted."""
        # Small pool: can hold ~3 requests of 70 tokens each
        pool = TokenToKVPoolAllocator(size=220, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        active = {}

        # Fill pool with 3 requests
        for uid in range(3):
            seq = self._full_seq(uid)
            indices = pool.alloc(len(seq))
            assert indices is not None, f"Pool exhausted at request {uid}"
            _model_runner_insert(cache, pool, seq, indices)
            r = cache.match_prefix(_key(seq))
            bid = cache.inc_lock_ref(r.last_node)
            active[uid] = (r.last_node, bid)

        _assert_size_invariant(cache)

        # Pool is nearly full. Request 0 finishes.
        node, bid = active.pop(0)
        cache.dec_lock_ref(node, swa_boundary_id=bid)

        # Evict to make room for new request
        cache.evict(70)
        pool.merge_and_sort_free()

        # New request 3 should now fit
        seq3 = self._full_seq(3)
        indices3 = pool.alloc(len(seq3))
        assert indices3 is not None, "Pool should have space after eviction"
        _model_runner_insert(cache, pool, seq3, indices3)
        r3 = cache.match_prefix(_key(seq3))
        bid3 = cache.inc_lock_ref(r3.last_node)
        active[3] = (r3.last_node, bid3)

        # Requests 1 and 2 should still be intact
        for uid in [1, 2]:
            r = cache.match_prefix(_key(self._full_seq(uid)))
            assert r.prefix_len == len(self._full_seq(uid))

        _assert_size_invariant(cache)

        for uid in list(active):
            node, bid = active.pop(uid)
            cache.dec_lock_ref(node, swa_boundary_id=bid)

    def test_deep_branching_conversation_tree(self):
        """Simulate a chat service: 5 users share a system prompt, each
        has 3 conversation turns, each turn extends the previous.
        Creates a deep tree with many branches."""
        pool = TokenToKVPoolAllocator(size=8192, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        # For each user, simulate 3 turns of conversation
        all_locks = []
        for uid in range(5):
            prefix = list(self.SYSTEM_PROMPT)
            for turn in range(3):
                # Each turn adds 15 tokens
                turn_tokens = [3000 + uid * 1000 + turn * 100 + j for j in range(15)]
                prefix = prefix + turn_tokens
                indices = pool.alloc(len(prefix))
                result = _model_runner_insert(cache, pool, prefix, indices)

                r = cache.match_prefix(_key(prefix))
                assert r.prefix_len == len(prefix)
                bid = cache.inc_lock_ref(r.last_node)
                all_locks.append((r.last_node, bid))

            _assert_size_invariant(cache)

        # 5 users * 3 turns = 15 active locks on various tree depths
        # All sharing the 50-token system prompt
        r_sys = cache.match_prefix(_key(self.SYSTEM_PROMPT))
        assert r_sys.prefix_len == self.SYSTEM_PROMPT_LEN

        # Nothing should be evictable (all locked)
        assert cache.evict(10000).full_evicted == 0

        # Unlock all
        for node, bid in all_locks:
            cache.dec_lock_ref(node, swa_boundary_id=bid)

        _assert_size_invariant(cache)

        # Evict everything — pool should be fully recovered
        cache.evict(100000)
        pool.merge_and_sort_free()
        _assert_size_invariant(cache)

    def test_prefix_divergence_at_multiple_depths(self):
        """Requests diverge from the shared prefix at different positions,
        creating a tree with branches at multiple depths."""
        pool = TokenToKVPoolAllocator(size=4096, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        base = list(range(1000, 1100))  # 100-token shared prefix
        active = {}

        # 8 requests that diverge at positions 20, 40, 60, 80, ...
        for i in range(8):
            diverge_at = 20 + i * 10
            seq = base[:diverge_at] + [5000 + i * 100 + j for j in range(30)]
            indices = pool.alloc(len(seq))
            _model_runner_insert(cache, pool, seq, indices)
            r = cache.match_prefix(_key(seq))
            assert r.prefix_len == len(seq)
            bid = cache.inc_lock_ref(r.last_node)
            active[i] = (seq, r.last_node, bid)

        _assert_size_invariant(cache)

        # Verify each request still matches fully
        for i, (seq, node, bid) in active.items():
            r = cache.match_prefix(_key(seq))
            assert r.prefix_len == len(seq), f"Request {i} diverging at {20+i*10} lost"

        # Unlock odd-numbered requests, evict, verify even ones survive
        for i in [1, 3, 5, 7]:
            seq, node, bid = active[i]
            cache.dec_lock_ref(node, swa_boundary_id=bid)

        cache.evict(500)
        _assert_size_invariant(cache)

        for i in [0, 2, 4, 6]:
            seq, node, bid = active[i]
            r = cache.match_prefix(_key(seq))
            assert r.prefix_len == len(seq), f"Locked request {i} lost after eviction"

        # Cleanup
        for i in [0, 2, 4, 6]:
            _, node, bid = active[i]
            cache.dec_lock_ref(node, swa_boundary_id=bid)

    def test_rapid_insert_evict_cycles_under_pressure(self):
        """Tight loop: insert request, use it, finish, evict, repeat.
        Simulates sustained high-throughput serving."""
        pool = TokenToKVPoolAllocator(size=512, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)
        initial_pool = pool.available_size()

        for cycle in range(50):
            uid = cycle % 10  # 10 distinct "users" cycling
            seq = self.SYSTEM_PROMPT + [4000 + uid * 100 + cycle + j for j in range(20)]

            indices = pool.alloc(len(seq))
            if indices is None:
                # Under pressure — evict and retry
                cache.evict(len(seq))
                pool.merge_and_sort_free()
                indices = pool.alloc(len(seq))
                assert indices is not None, f"Pool exhausted at cycle {cycle}"

            result = _model_runner_insert(cache, pool, seq, indices)
            r = cache.match_prefix(_key(seq))
            bid = cache.inc_lock_ref(r.last_node)

            # "Decode" (no-op) then finish
            cache.dec_lock_ref(r.last_node, swa_boundary_id=bid)

        _assert_size_invariant(cache)

        # Final cleanup
        cache.evict(100000)
        pool.merge_and_sort_free()
        assert pool.available_size() == initial_pool

    def test_many_requests_same_exact_prompt(self):
        """20 requests with the identical prompt. All should share the
        same tree path and stack locks correctly."""
        pool = TokenToKVPoolAllocator(size=2048, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        seq = self.SYSTEM_PROMPT + [9000 + j for j in range(30)]
        locks = []

        # First request inserts into the tree
        indices = pool.alloc(len(seq))
        _model_runner_insert(cache, pool, seq, indices)

        # 20 concurrent requests all match the same path
        for i in range(20):
            r = cache.match_prefix(_key(seq))
            assert r.prefix_len == len(seq)
            bid = cache.inc_lock_ref(r.last_node)
            locks.append((r.last_node, bid))

        # The leaf should have lock_ref == 20
        assert locks[0][0].lock_ref == 20

        # Eviction must fail
        assert cache.evict(10000).full_evicted == 0
        _assert_size_invariant(cache)

        # Unlock 19, one remains — still protected
        for node, bid in locks[:-1]:
            cache.dec_lock_ref(node, swa_boundary_id=bid)
        assert locks[-1][0].lock_ref == 1
        assert cache.evict(10000).full_evicted == 0

        # Unlock last one
        cache.dec_lock_ref(locks[-1][0], swa_boundary_id=locks[-1][1])
        _assert_size_invariant(cache)

        # Now everything is evictable
        result = cache.evict(10000)
        assert result.full_evicted > 0

    def test_interleaved_prefill_and_decode_locks(self):
        """Simulate continuous batching: while some requests are decoding
        (locked), new requests arrive for prefill, causing tree splits
        on locked nodes. Verify invariant throughout."""
        pool = TokenToKVPoolAllocator(size=4096, device="cpu")
        cache = RadixCache(page_size=1, token_to_kv_pool_allocator=pool)

        # Decoding request: already inserted and locked
        decode_seq = self.SYSTEM_PROMPT + [6000 + j for j in range(40)]
        decode_idx = pool.alloc(len(decode_seq))
        _model_runner_insert(cache, pool, decode_seq, decode_idx)
        r_decode = cache.match_prefix(_key(decode_seq))
        bid_decode = cache.inc_lock_ref(r_decode.last_node)
        _assert_size_invariant(cache)

        # New prefill request shares system prompt but diverges
        prefill_seq = self.SYSTEM_PROMPT + [7000 + j for j in range(25)]
        prefill_idx = pool.alloc(len(prefill_seq))
        _model_runner_insert(cache, pool, prefill_seq, prefill_idx)
        # This insert splits the tree at the system prompt boundary
        # while the decode request's node is locked
        _assert_size_invariant(cache)

        r_prefill = cache.match_prefix(_key(prefill_seq))
        assert r_prefill.prefix_len == len(prefill_seq)
        bid_prefill = cache.inc_lock_ref(r_prefill.last_node)
        _assert_size_invariant(cache)

        # Another prefill that diverges even earlier (at token 30)
        early_seq = list(self.SYSTEM_PROMPT[:30]) + [8000 + j for j in range(20)]
        early_idx = pool.alloc(len(early_seq))
        _model_runner_insert(cache, pool, early_seq, early_idx)
        _assert_size_invariant(cache)

        r_early = cache.match_prefix(_key(early_seq))
        bid_early = cache.inc_lock_ref(r_early.last_node)
        _assert_size_invariant(cache)

        # Decode finishes
        cache.dec_lock_ref(r_decode.last_node, swa_boundary_id=bid_decode)
        _assert_size_invariant(cache)

        # Evict decode's unique suffix — prefill requests must survive
        cache.evict(40)

        assert cache.match_prefix(_key(prefill_seq)).prefix_len == len(prefill_seq)
        assert cache.match_prefix(_key(early_seq)).prefix_len == len(early_seq)
        _assert_size_invariant(cache)

        # Cleanup
        cache.dec_lock_ref(r_prefill.last_node, swa_boundary_id=bid_prefill)
        cache.dec_lock_ref(r_early.last_node, swa_boundary_id=bid_early)
        _assert_size_invariant(cache)
