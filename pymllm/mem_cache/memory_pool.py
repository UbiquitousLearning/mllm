"""Lightweight KV-cache memory pools

Three-layer architecture::

    ReqToTokenPool          maps  (req_slot, position) → kv_index
    TokenToKVPoolAllocator  manages a free-list of integer indices
    KVPool                  holds the actual GPU K/V tensors

All indices are **int64** tensors on the target device.  Slot 0 in the KV
buffers is reserved as a padding / dummy-output slot and is never allocated.
"""

import logging
from typing import List, Optional, Tuple, Union

import torch

from mllm_kernel.cuda.jit.store_cache import store_cache, can_use_store_cache

logger = logging.getLogger(__name__)


class KVPool:
    """GPU (or CPU) storage for per-layer key and value caches.

    Layout per layer::

    JIT:
        k_buffer[layer][slot, k_head_num * k_head_dim]
        v_buffer[layer][slot, v_head_num * v_head_dim]

    PyTorch:
        k_buffer[layer][slot, k_head_num, k_head_dim]
        v_buffer[layer][slot, v_head_num, v_head_dim]

    K and V may have **independent** head counts and head dimensions, which
    covers standard MHA, GQA / MQA, and architectures like MLA where value
    projection uses a different dimensionality.

    ``size`` usable slots are numbered ``[1, size]``.  Slot 0 is a dummy
    padding slot that absorbs writes from padded tokens.

    Parameters
    ----------
    size : int
        Number of usable token slots (total buffer length = ``size + 1``).
    layer_num : int
        Number of transformer layers (one K buffer + one V buffer per layer).
    k_head_num : int
        Number of key heads.
    k_head_dim : int
        Dimension of each key head.
    device : str | torch.device
        Target device (``"cuda"``, ``"cpu"``, …).
    dtype : torch.dtype
        Storage data type.
    v_head_num : int, optional
        Number of value heads.  Defaults to *k_head_num*.
    v_head_dim : int, optional
        Dimension of each value head.  Defaults to *k_head_dim*.
    pin_memory : bool, optional
        Whether to use pinned memory.  Defaults to True.
    """

    def __init__(
        self,
        size: int,
        layer_num: int,
        k_head_num: int,
        k_head_dim: int,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
        v_head_num: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        pin_memory: bool = True,
    ):
        self.size = size
        self.layer_num = layer_num
        self.k_head_num = k_head_num
        self.k_head_dim = k_head_dim
        self.v_head_num = v_head_num if v_head_num is not None else k_head_num
        self.v_head_dim = v_head_dim if v_head_dim is not None else k_head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        buf_len = size + 1  # slot 0 is padding

        if buf_len % 8 != 0:
            logger.warning(
                "KVPool buffer length is not divisible by 8, padding to the next multiple of 8"
            )
            buf_len = (buf_len + 7) & ~7

        k_row_dim = self.k_head_num * self.k_head_dim
        v_row_dim = self.v_head_num * self.v_head_dim
        self._same_kv_dim = k_row_dim == v_row_dim
        self._row_bytes = k_row_dim * torch.tensor([], dtype=dtype).element_size()
        self._use_jit = (
            self.device.type == "cuda"
            and self._same_kv_dim
            and can_use_store_cache(self._row_bytes)
        )
        if not self._use_jit:
            logger.warning(
                f"Fallback to PyTorch index for KVPool, which is slower than the mllm-kernel's implementation, same_kv_dim={self._same_kv_dim}, row_bytes={self._row_bytes}"
            )

        self.k_buffer: List[torch.Tensor] = [
            torch.zeros(
                (buf_len, self.k_head_num, self.k_head_dim),
                dtype=dtype,
                device=self.device,
                pin_memory=pin_memory,
            )
            for _ in range(layer_num)
        ]
        self.v_buffer: List[torch.Tensor] = [
            torch.zeros(
                (buf_len, self.v_head_num, self.v_head_dim),
                dtype=dtype,
                device=self.device,
                pin_memory=pin_memory,
            )
            for _ in range(layer_num)
        ]

        # Pre-computed 2D views for the JIT store_cache kernel.
        # Zero-copy: same underlying storage as k_buffer / v_buffer.
        if self._use_jit:
            self._k_buffer_2d = [b.view(buf_len, -1) for b in self.k_buffer]
            self._v_buffer_2d = [b.view(buf_len, -1) for b in self.v_buffer]

        logger.info(
            "KVPool allocated: %d layers, %d slots, K=[%d,%d] V=[%d,%d], %.2f GB",
            layer_num,
            size,
            self.k_head_num,
            self.k_head_dim,
            self.v_head_num,
            self.v_head_dim,
            self._mem_bytes() / (1 << 30),
        )

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_buffer[layer_id], self.v_buffer[layer_id]

    def set_kv_buffer(
        self,
        layer_id: int,
        indices: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        """Write K/V vectors into the cache at the given *indices*.

        ``k`` / ``v`` can be any shape as long as the trailing dimensions
        multiply to ``head_num * head_dim`` (the row dimension).  All leading
        dimensions are treated as the batch axis and must match ``indices``
        after flattening.  Typical shapes::

            k: [num_tokens, head_num, head_dim]          indices: [num_tokens]
            k: [batch, seq_len, head_num, head_dim]      indices: [batch, seq_len]
            k: [num_tokens, head_num * head_dim]          indices: [num_tokens]
        """
        if self._use_jit:
            row_dim = self.k_head_num * self.k_head_dim
            store_cache(
                k.reshape(-1, row_dim),
                v.reshape(-1, row_dim),
                self._k_buffer_2d[layer_id],
                self._v_buffer_2d[layer_id],
                indices.reshape(-1),
                row_bytes=self._row_bytes,
            )
        else:
            self.k_buffer[layer_id][indices] = k
            self.v_buffer[layer_id][indices] = v

    def _mem_bytes(self) -> int:
        total = 0
        for buf in self.k_buffer + self.v_buffer:
            total += buf.nelement() * buf.element_size()
        return total


class TokenToKVPoolAllocator:
    """Manages allocation / deallocation of integer indices into a :class:`KVPool`.

    Each ``alloc(n)`` returns *n* free indices; each ``free(indices)`` returns
    them to the pool.

    Uses a **dual-buffer** strategy (``free_slots`` + ``release_slots``) so
    that ``free()`` never cats onto the large main free-list.  Freed indices
    accumulate in the smaller ``release_slots`` and are merged lazily (with an
    optional sort) only when ``alloc()`` cannot be satisfied from
    ``free_slots`` alone.

    A **batch-free** API (``free_group_begin`` / ``free_group_end``) further
    amortises cost when many ``free()`` calls happen in a tight loop (e.g.
    during scheduling or eviction).

    Typical usage::

        allocator = TokenToKVPoolAllocator(size=4096, device="cuda")

        # --- basic alloc / free ---
        indices = allocator.alloc(128)      # 128 free slot indices (int64)
        allocator.free(indices[:64])        # return 64 slots

        # --- batch free (amortised) ---
        allocator.free_group_begin()
        for req in finished_requests:
            allocator.free(req.kv_indices)  # O(1) list append each
        allocator.free_group_end()          # single torch.cat + release

    Parameters
    ----------
    size : int
        Total number of allocatable slots (must match ``KVPool.size``).
    device : str | torch.device
        Device for the free-list tensor.
    page_size : int
        When > 1 the allocator works in page-aligned mode: ``alloc`` returns
        multiples of ``page_size`` contiguous within each page, and ``free``
        deduplicates by page.
    need_sort : bool
        When ``True`` (default), ``merge_and_sort_free`` sorts after merging
        so that lower-index slots are allocated first (better memory locality).
    """

    def __init__(
        self,
        size: int,
        device: Union[str, torch.device] = "cuda",
        page_size: int = 1,
        need_sort: bool = True,
    ):
        self.size = size
        self.page_size = page_size
        self.device = torch.device(device)
        self.need_sort = need_sort
        self.clear()

    def clear(self) -> None:
        """Reset the allocator so that all slots ``[1, size]`` are free. The first slot is reserved for padding."""
        if self.page_size == 1:
            self.free_slots = torch.arange(
                1, self.size + 1, dtype=torch.int64, device=self.device
            )
        else:
            num_pages = self.size // self.page_size
            self.free_slots = torch.arange(
                1, num_pages + 1, dtype=torch.int64, device=self.device
            )
        self.release_slots = torch.empty((0,), dtype=torch.int64, device=self.device)
        self._is_not_in_free_group = True
        self._free_group: List[torch.Tensor] = []

    def available_size(self) -> int:
        """Number of tokens that can still be allocated."""
        return (len(self.free_slots) + len(self.release_slots)) * self.page_size

    def merge_and_sort_free(self) -> None:
        """Merge ``release_slots`` into ``free_slots`` (and sort if ``need_sort``)."""
        if len(self.release_slots) == 0:
            return
        self.free_slots = torch.cat((self.free_slots, self.release_slots))
        if self.need_sort:
            self.free_slots, _ = torch.sort(self.free_slots)
        self.release_slots = torch.empty((0,), dtype=torch.int64, device=self.device)

    def free_group_begin(self) -> None:
        """Start collecting ``free()`` calls; actual release is deferred to ``free_group_end``."""
        self._is_not_in_free_group = False
        self._free_group = []

    def free_group_end(self) -> None:
        """Flush all ``free()`` calls collected since ``free_group_begin``."""
        self._is_not_in_free_group = True
        if self._free_group:
            self.free(torch.cat(self._free_group))
            self._free_group = []

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        """Allocate *need_size* token indices.

        Returns a 1-D ``int64`` tensor on success, or ``None`` if the pool is
        exhausted.
        """
        if self.page_size == 1:
            if need_size > len(self.free_slots):
                self.merge_and_sort_free()
            if need_size > len(self.free_slots):
                return None
            out = self.free_slots[:need_size]
            self.free_slots = self.free_slots[need_size:]
            return out

        num_pages = (need_size + self.page_size - 1) // self.page_size
        if num_pages > len(self.free_slots):
            self.merge_and_sort_free()
        if num_pages > len(self.free_slots):
            return None
        pages = self.free_slots[:num_pages]
        self.free_slots = self.free_slots[num_pages:]
        offsets = torch.arange(self.page_size, device=self.device)
        out = (pages[:, None] * self.page_size + offsets).reshape(-1)
        return out[:need_size]

    def free(self, indices: torch.Tensor) -> None:
        """Return *indices* to the free pool."""
        if indices.numel() == 0:
            return

        if not self._is_not_in_free_group:
            self._free_group.append(indices)
            return

        if self.page_size != 1:
            indices = torch.unique(indices // self.page_size)

        if self.need_sort:
            self.release_slots = torch.cat((self.release_slots, indices))
        else:
            self.free_slots = torch.cat((self.free_slots, indices))


class ReqToTokenPool:
    """Maps each live request to its per-position KV-pool indices.

    Internally a 2-D tensor ``req_to_token[slot, position]`` stores the
    KV-pool index for every token position of every active request.
    Slots are recycled via a simple free-list.

    This class is a **pure mapping table** -- it does **not** track per-request
    sequence lengths.  The caller (typically the ``Req`` / IO-struct object)
    must store ``req_pool_idx`` and ``seq_len`` and use them to slice into
    ``req_to_token`` when reading back KV indices.

    Typical usage::

        pool = ReqToTokenPool(max_reqs=256, max_context_len=4096)

        # --- on new request arrival ---
        [slot] = pool.alloc(1)                       # slot = req_pool_idx
        kv_indices = kv_allocator.alloc(seq_len)      # from TokenToKVPoolAllocator
        pool.write((slot, slice(0, seq_len)), kv_indices)

        # --- read back (caller tracks seq_len) ---
        kv_indices = pool.req_to_token[slot, :seq_len]

        # --- on request completion ---
        kv_allocator.free(pool.req_to_token[slot, :seq_len])
        pool.free(slot)

    Parameters
    ----------
    max_reqs : int
        Maximum number of concurrent requests (number of rows).
    max_context_len : int
        Maximum sequence length any single request can reach (number of cols).
    device : str | torch.device
        Target device for the mapping tensor.
    """

    def __init__(
        self,
        max_reqs: int,
        max_context_len: int,
        device: Union[str, torch.device] = "cuda",
    ):
        self.size = max_reqs
        self.max_context_len = max_context_len
        self.device = torch.device(device)

        self.req_to_token = torch.zeros(
            (max_reqs, max_context_len), dtype=torch.int64, device=self.device
        )
        self._free_slots: List[int] = list(range(max_reqs))

    def available_size(self) -> int:
        return len(self._free_slots)

    def alloc(self, n: int = 1) -> Optional[List[int]]:
        """Allocate *n* request slots.  Returns a list of slot indices."""
        if n > len(self._free_slots):
            return None
        out = self._free_slots[:n]
        self._free_slots = self._free_slots[n:]
        return out

    def free(self, slot: int) -> None:
        """Return a single request slot to the pool."""
        self._free_slots.append(slot)

    def write(self, index: Tuple, values: torch.Tensor) -> None:
        """Write KV indices into the mapping table.

        ``index`` is typically ``(req_pool_idx, slice(start, end))``.
        """
        self.req_to_token[index] = values

    def clear(self) -> None:
        self._free_slots = list(range(self.size))
        self.req_to_token.zero_()


def make_full_attention_net_mem_pool(
    size: int,
    layer_num: int,
    k_head_num: int,
    k_head_dim: int,
    v_head_num: int,
    v_head_dim: int,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float16,
    page_size: int = 1,
    need_sort: bool = True,
    pin_memory: bool = True,
) -> Tuple[KVPool, TokenToKVPoolAllocator]:
    """Create a :class:`KVPool` and its :class:`TokenToKVPoolAllocator` for a
    full-attention (non-SWA) model.

    Parameters
    ----------
    size : int
        Number of usable token slots in the KV cache.
    layer_num : int
        Number of transformer layers.
    k_head_num / k_head_dim : int
        Key head count and dimension.
    v_head_num / v_head_dim : int
        Value head count and dimension.
    device : str | torch.device
        Target device.
    dtype : torch.dtype
        Storage data type for the KV buffers.
    page_size : int
        Allocator page size (1 = per-token, >1 = page-aligned).
    need_sort : bool
        Whether the allocator sorts on merge for memory locality.
    pin_memory : bool
        Whether to use pinned memory for the KV buffers.

    Returns
    -------
    (KVPool, TokenToKVPoolAllocator)
    """
    pool = KVPool(
        size=size,
        layer_num=layer_num,
        k_head_num=k_head_num,
        k_head_dim=k_head_dim,
        device=device,
        dtype=dtype,
        v_head_num=v_head_num,
        v_head_dim=v_head_dim,
        pin_memory=pin_memory,
    )
    allocator = TokenToKVPoolAllocator(
        size=size,
        device=device,
        page_size=page_size,
        need_sort=need_sort,
    )
    return pool, allocator


def make_req_to_token_pool(
    max_reqs: int,
    max_context_len: int,
    device: Union[str, torch.device] = "cuda",
) -> ReqToTokenPool:
    return ReqToTokenPool(max_reqs, max_context_len, device)
