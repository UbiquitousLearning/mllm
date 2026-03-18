"""
CUDA IPC Transport for zero-copy GPU tensor sharing between processes.

## Background

When sharing CUDA tensors between processes, there are two fundamentally different paths:

1. **CPU shared memory path** (``enable_shared_queue=True, enable_cuda_ipc=False``):
   GPU tensors are moved to CPU / POSIX shared memory via ``tensor.share_memory_()``.
   This is safe but incurs a GPU→CPU copy which is expensive for large vision features.

2. **CUDA IPC path** (``enable_cuda_ipc=True``):
   GPU tensors stay on GPU. PyTorch's ``storage._share_cuda_()`` yields a serialisable
   IPC handle; the receiver calls ``UntypedStorage._new_shared_cuda(*handle)`` to map
   the same physical GPU memory without any copy.

These two paths are **mutually exclusive for GPU tensors**. ``enable_cuda_ipc`` takes
priority; when active the CPU-copy step in ``TensorQueue._make_tensors_shareable`` is
skipped.

## CUDA IPC memory-leak problem and its fix

PyTorch never releases the GPU allocation backing an IPC-exported tensor until the
*sending* process exits. If we export raw model tensors we permanently leak GPU memory.

**Solution** (pool-based recycling via ``MmItemMemoryPool``):

* Allocate a single, fixed-size GPU workspace (``MmItemMemoryPool``).
* For each outgoing GPU tensor, copy it into a chunk of the workspace and export the
  *chunk* via IPC (the workspace is never freed; its chunks are recycled).
* After the receiving process has finished with the data it writes a sync flag
  (``ShmSyncBuffer``) to signal that the chunk may be reused.
* A background recycler thread in the sender walks ``occupied_chunks`` and returns
  chunks whose sync flag has been incremented back to ``available_chunks``.

## Transport modes

``TensorTransportMode``:
* ``"default"``  – CPU/shared-memory path; no CUDA IPC.
* ``"cuda_ipc"`` – Simple CUDA IPC: wraps GPU tensors in ``TransportProxyTensor``
  (a ``torch.Tensor`` subclass whose ``__getstate__``/``__setstate__`` use
  ``_share_cuda_``).  Suitable for single-process-group scenarios; incurs the
  PyTorch memory-leak noted above.
* ``"cuda_ipc_pool"`` – Pool-based CUDA IPC: copies GPU tensors into a pre-allocated
  ``MmItemMemoryPool`` and wraps the slice in ``CudaIpcTensorTransportProxy``.
  The pool is recycled, so there is no memory leak.
"""

from __future__ import annotations

import fcntl
import logging
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for transport mode
# ---------------------------------------------------------------------------

TensorTransportMode = Literal["default", "cuda_ipc", "cuda_ipc_pool"]


# ---------------------------------------------------------------------------
# ShmSyncBuffer – a tiny POSIX shared memory float used as a sync counter
# ---------------------------------------------------------------------------


class ShmSyncBuffer:
    """A single float32 in POSIX shared memory used as a sync counter.

    The sender resets it to 0 before exporting a chunk.  The receiver
    increments it (atomically under a file lock) once it has finished copying
    data out of the chunk.  When the value reaches the number of consumers
    (``tp_size``) the sender recycles the chunk.
    """

    def __init__(self, byte_size: int = 4) -> None:
        self.buffer = shared_memory.SharedMemory(create=True, size=byte_size)
        self._arr = np.ndarray(1, dtype=np.float32, buffer=self.buffer.buf)
        self._arr *= 0  # initialise to 0
        self.meta_data: Dict[str, Any] = {
            "handle": self.buffer.name,
            "shape": self._arr.shape,
            "dtype": str(self._arr.dtype),
        }

    # ------------------------------------------------------------------
    # Helpers consumed by the *receiver* side
    # ------------------------------------------------------------------

    @staticmethod
    def open(
        meta_data: Dict[str, Any],
    ) -> Tuple[shared_memory.SharedMemory, np.ndarray]:
        """Open an existing ShmSyncBuffer from the metadata dict."""
        shm = shared_memory.SharedMemory(name=meta_data["handle"])
        arr = np.ndarray(meta_data["shape"], dtype=meta_data["dtype"], buffer=shm.buf)
        return shm, arr

    def __del__(self) -> None:
        try:
            self.buffer.close()
            self.buffer.unlink()
        except Exception:
            pass


# Lock file used to serialise writes to sync flags across processes
_SHM_LOCK_FILE = "/tmp/pymllm_shm_wr_lock.lock"


def _increment_sync_flag(meta_data: Dict[str, Any]) -> None:
    """Increment the sync flag by 1 under a process-level file lock."""
    shm, arr = ShmSyncBuffer.open(meta_data)
    try:
        open(_SHM_LOCK_FILE, "a").close()  # ensure file exists
        with open(_SHM_LOCK_FILE, "w+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            arr += 1.0
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        shm.close()


# ---------------------------------------------------------------------------
# MmItemMemoryChunk
# ---------------------------------------------------------------------------


class MmItemMemoryChunk:
    """A contiguous slice of the ``MmItemMemoryPool`` workspace tensor."""

    def __init__(self, area: Tuple[int, int], sync_flag: ShmSyncBuffer) -> None:
        self.area = area
        self.sync_flag = sync_flag

    @property
    def mem_size(self) -> int:
        return self.area[1] - self.area[0]

    @property
    def start(self) -> int:
        return self.area[0]

    @property
    def end(self) -> int:
        return self.area[1]

    def try_to_recycle(self, num_consumers: int = 1) -> bool:
        """Return True if all consumers have finished and the chunk can be reused."""
        val = float(self.sync_flag._arr.item())
        logger.debug(
            "[try_to_recycle] area=%s flag=%.0f consumers=%d",
            self.area,
            val,
            num_consumers,
        )
        if val >= float(num_consumers):
            self.sync_flag._arr *= 0.0  # reset for next use
            return True
        return False


# ---------------------------------------------------------------------------
# MmItemMemoryPool – pre-allocated GPU workspace to avoid IPC memory leaks
# ---------------------------------------------------------------------------


class MmItemMemoryPool:
    """Pre-allocated GPU memory pool for CUDA IPC tensor transport.

    Chunks are allocated from a contiguous ``torch.int8`` tensor on GPU.
    A background thread periodically recycles chunks whose sync flags show
    that all consumers have finished reading.

    Args:
        memory_size: Pool size in **bytes**.
        recycle_interval: How often (seconds) the recycler thread runs.
        num_consumers: Number of consumer processes (tp_size). Each consumer
            must increment the sync flag once before a chunk is recycled.
        device: CUDA device index.
    """

    def __init__(
        self,
        memory_size: int,
        recycle_interval: float = 0.1,
        num_consumers: int = 1,
        device: int = 0,
    ) -> None:
        self.num_consumers = num_consumers
        self._recycle_interval = recycle_interval
        self._lock = threading.Lock()
        self._stop = False

        with torch.cuda.device(device):
            self.memory_pool: torch.Tensor = torch.empty(
                memory_size, dtype=torch.int8, device=f"cuda:{device}"
            ).contiguous()

        init_chunk = MmItemMemoryChunk((0, memory_size), self._new_sync_buffer())
        self.available_chunks: List[MmItemMemoryChunk] = [init_chunk]
        self.occupied_chunks: List[MmItemMemoryChunk] = []
        # Pool of reusable ShmSyncBuffer objects (returned from recycled chunks)
        self._sync_pool: List[ShmSyncBuffer] = []

        self._recycler = threading.Thread(
            target=self._recycle_loop,
            name="MmItemMemoryPoolRecycler",
            daemon=True,
        )
        self._recycler.start()

        logger.info(
            "MmItemMemoryPool: %d MB on cuda:%d, recycle_interval=%.2fs",
            memory_size // (1024 * 1024),
            device,
            recycle_interval,
        )

    # ------------------------------------------------------------------
    # Sync buffer management
    # ------------------------------------------------------------------

    def _new_sync_buffer(self) -> ShmSyncBuffer:
        if self._sync_pool:
            return self._sync_pool.pop()
        return ShmSyncBuffer()

    def _return_sync_buffer(self, buf: ShmSyncBuffer) -> None:
        buf._arr *= 0.0  # reset counter
        self._sync_pool.append(buf)

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def _get_available_chunk(self, src: torch.Tensor) -> Optional[MmItemMemoryChunk]:
        """Best-fit allocation: find the smallest available chunk >= src size."""
        needed = src.numel() * src.element_size()
        best: Optional[MmItemMemoryChunk] = None
        for chunk in self.available_chunks:
            if chunk.mem_size >= needed:
                if best is None or chunk.mem_size < best.mem_size:
                    best = chunk
        if best is None:
            return None

        # Split the selected chunk
        occupied_area = (best.start, best.start + needed)
        occupied = MmItemMemoryChunk(occupied_area, best.sync_flag)
        self.occupied_chunks.append(occupied)
        self.available_chunks.remove(best)

        remainder = (occupied.end, best.end)
        if remainder[0] < remainder[1]:
            split = MmItemMemoryChunk(remainder, self._new_sync_buffer())
            self.available_chunks.append(split)

        return occupied

    def get_slice_with_flag(
        self, src: torch.Tensor
    ) -> Tuple[Optional[Dict[str, Any]], Optional[torch.Tensor]]:
        """Allocate a pool slice for *src* and return ``(sync_flag_meta, slice_tensor)``.

        Thread-safe.  Returns ``(None, None)`` if the pool is full.
        """
        with self._lock:
            chunk = self._get_available_chunk(src)
            if chunk is None:
                logger.warning(
                    "MmItemMemoryPool full (%d occupied, %d available); "
                    "falling back to CPU transport",
                    len(self.occupied_chunks),
                    len(self.available_chunks),
                )
                return None, None
            pool_slice = self.memory_pool[chunk.start : chunk.end]
            return chunk.sync_flag.meta_data, pool_slice

    # ------------------------------------------------------------------
    # Recycling
    # ------------------------------------------------------------------

    def _recycle_loop(self) -> None:
        while not self._stop:
            try:
                with self._lock:
                    self._recycle_chunks()
                    self._merge_chunks()
            except Exception as exc:
                logger.warning(
                    "MmItemMemoryPool recycler error: %s", exc, exc_info=True
                )
            time.sleep(self._recycle_interval)

    def _recycle_chunks(self) -> None:
        new_occupied: List[MmItemMemoryChunk] = []
        for chunk in self.occupied_chunks:
            if chunk.try_to_recycle(self.num_consumers):
                self._return_sync_buffer(chunk.sync_flag)
                chunk.sync_flag = self._new_sync_buffer()
                self.available_chunks.append(chunk)
            else:
                new_occupied.append(chunk)
        self.occupied_chunks = new_occupied

    def _merge_chunks(self) -> None:
        """Coalesce adjacent free chunks to reduce fragmentation."""
        merged: List[MmItemMemoryChunk] = []
        for chunk in sorted(self.available_chunks, key=lambda c: c.start):
            if merged and merged[-1].end == chunk.start:
                prev = merged.pop()
                self._return_sync_buffer(chunk.sync_flag)
                merged.append(
                    MmItemMemoryChunk((prev.start, chunk.end), prev.sync_flag)
                )
            else:
                merged.append(chunk)
        self.available_chunks = merged

    def shutdown(self) -> None:
        self._stop = True
        if self._recycler.is_alive():
            self._recycler.join(timeout=2.0)


# ---------------------------------------------------------------------------
# CudaIpcTensorTransportProxy – pool-based CUDA IPC proxy object
# ---------------------------------------------------------------------------


class CudaIpcTensorTransportProxy:
    """Proxy that carries a CUDA IPC handle for a pool-slice tensor.

    The *sender* process:
    1. Copies the source tensor into a ``MmItemMemoryPool`` slice (int8 view).
    2. Wraps the slice in this proxy, which captures the CUDA IPC handle via
       ``storage._share_cuda_()``.
    3. Sends the proxy through ``multiprocessing.Queue`` (pickle).

    The *receiver* process:
    1. Calls :meth:`reconstruct_on_device` to map the IPC memory and copy it
       into a fresh local tensor.
    2. The copy increments the sync flag, allowing the sender's recycler to
       reclaim the pool slice.

    Fallback: if ``_share_cuda_()`` fails (e.g. TP ranks), ``tensor_data`` holds
    the raw tensor (which will be pickled the normal way, incurring serialization cost).
    """

    def __init__(
        self,
        data: torch.Tensor,
        info_data: torch.Tensor,
        sync_buffer_meta: Dict[str, Any],
    ) -> None:
        if not isinstance(data, torch.Tensor) or not isinstance(
            info_data, torch.Tensor
        ):
            raise TypeError(
                f"data and info_data must be torch.Tensors, got {type(data)}, {type(info_data)}"
            )

        self.sync_data_meta = sync_buffer_meta
        self._state = self._build_state(data, info_data)
        self._reconstructed: Optional[torch.Tensor] = None
        self._shm: Optional[shared_memory.SharedMemory] = None

    def _build_state(
        self, data: torch.Tensor, info_data: torch.Tensor
    ) -> Dict[str, Any]:
        try:
            storage = data.untyped_storage()
            handle = storage._share_cuda_()
            return {
                "ipc_handle": {
                    "handle": handle,
                    "shape": data.shape,
                    "dtype": data.dtype,
                    "stride": data.stride(),
                    "device_index": data.device.index,
                    "storage_offset": data.storage_offset(),
                    "target_shape": info_data.shape,
                    "target_dtype": info_data.dtype,
                },
                "tensor_data": None,
            }
        except Exception as exc:
            logger.warning(
                "CudaIpcTensorTransportProxy: _share_cuda_() failed (%s); "
                "falling back to direct tensor.",
                exc,
            )
            return {"ipc_handle": None, "tensor_data": data}

    def reconstruct_on_device(self, device_index: Optional[int] = None) -> torch.Tensor:
        """Map IPC memory and copy into a new local tensor.

        This **must** be called from the *receiver* process.  After the copy
        the sync flag is incremented so the sender can recycle the pool chunk.
        """
        if self._reconstructed is not None:
            return self._reconstructed

        state = self._state
        if state["ipc_handle"] is not None:
            h = state["ipc_handle"]
            source_device = torch.device(f"cuda:{h['device_index']}")
            target_device = (
                source_device
                if device_index is None
                else torch.device(f"cuda:{device_index}")
            )
            with torch.cuda.device(source_device):
                storage = torch.UntypedStorage._new_shared_cuda(*h["handle"])
                slice_tensor = torch.empty(
                    0, dtype=h["dtype"], device=source_device
                ).set_(
                    storage,
                    storage_offset=h["storage_offset"],
                    size=h["shape"],
                    stride=h["stride"],
                )

                result = torch.empty(
                    h["target_shape"], dtype=h["target_dtype"], device=target_device
                ).contiguous()
                result.view(torch.int8).view(-1).copy_(slice_tensor)

            # Signal sender that the chunk can be recycled
            _increment_sync_flag(self.sync_data_meta)
        elif state["tensor_data"] is not None:
            result = state["tensor_data"]
            if device_index is not None:
                result = result.to(f"cuda:{device_index}", non_blocking=True)
        else:
            raise RuntimeError("CudaIpcTensorTransportProxy: invalid state")

        self._reconstructed = result
        return result


# ---------------------------------------------------------------------------
# TransportProxyTensor – simple CUDA IPC via torch.Tensor subclass + pickle
# ---------------------------------------------------------------------------


class TransportProxyTensor(torch.Tensor):
    """A ``torch.Tensor`` subclass whose pickle uses CUDA IPC handles.

    When ``transport_mode == "cuda_ipc"`` and the tensor is on CUDA,
    ``__getstate__`` exports the tensor via ``storage._share_cuda_()`` instead
    of serialising the raw data.  ``__setstate__`` reconstructs it in the
    receiving process via ``UntypedStorage._new_shared_cuda``.

    Caveat: The underlying GPU allocation is never freed until the *sender*
    process exits (PyTorch limitation).  Prefer ``"cuda_ipc_pool"`` mode for
    long-running services to avoid GPU memory leaks.

    When the tensor is on CPU or ``transport_mode == "default"``, the tensor
    is serialised normally (pickle of raw data).
    """

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        transport_mode: TensorTransportMode = "default",
    ) -> "TransportProxyTensor":
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"data must be a torch.Tensor, got {type(data)}")
        instance = data.as_subclass(cls)
        instance._transport_mode = transport_mode
        return instance

    def __getstate__(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "transport_mode": self._transport_mode,
            "tensor_data": None,
            "ipc_extra": None,
        }
        if self._transport_mode == "cuda_ipc" and self.is_cuda:
            try:
                storage = self.untyped_storage()
                handle = storage._share_cuda_()
                state["ipc_extra"] = {
                    "handle": handle,
                    "shape": self.shape,
                    "dtype": self.dtype,
                    "stride": self.stride(),
                    "device_index": self.device.index,
                    "storage_offset": self.storage_offset(),
                }
            except Exception as exc:
                logger.warning(
                    "TransportProxyTensor: _share_cuda_() failed (%s); falling back.",
                    exc,
                )
                state["transport_mode"] = "default"
                state["tensor_data"] = self.as_subclass(torch.Tensor)
        else:
            state["transport_mode"] = "default"
            state["tensor_data"] = self.as_subclass(torch.Tensor)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self._transport_mode = state["transport_mode"]
        if state["transport_mode"] == "cuda_ipc" and state["ipc_extra"] is not None:
            h = state["ipc_extra"]
            target = torch.device(f"cuda:{h['device_index']}")
            try:
                with torch.cuda.device(target):
                    storage = torch.UntypedStorage._new_shared_cuda(*h["handle"])
                    reconstructed = torch.empty(
                        0, dtype=h["dtype"], device=target
                    ).set_(
                        storage,
                        storage_offset=h["storage_offset"],
                        size=h["shape"],
                        stride=h["stride"],
                    )
                    self.set_(reconstructed)
            except Exception as exc:
                logger.error("TransportProxyTensor: failed to open IPC handle: %s", exc)
                raise
        elif state["tensor_data"] is not None:
            self.set_(state["tensor_data"])
        else:
            raise RuntimeError("TransportProxyTensor: invalid state – no tensor data")

    @property
    def transport_mode(self) -> TensorTransportMode:
        return getattr(self, "_transport_mode", "default")


# ---------------------------------------------------------------------------
# Helpers: wrap / unwrap mm_inputs dicts
# ---------------------------------------------------------------------------


def wrap_mm_inputs_for_ipc(
    mm_inputs: Optional[Dict[str, Any]],
    transport_mode: TensorTransportMode,
    pool: Optional["MmItemMemoryPool"] = None,
) -> Optional[Dict[str, Any]]:
    """Recursively wrap CUDA tensors in *mm_inputs* for IPC transport.

    Args:
        mm_inputs: Nested dict/list of tensors and other data.
        transport_mode: One of ``"default"``, ``"cuda_ipc"``, ``"cuda_ipc_pool"``.
        pool: Required when ``transport_mode == "cuda_ipc_pool"``.

    Returns:
        A new data structure with CUDA tensors replaced by IPC proxies.
        CPU tensors are left unchanged (they will be shared via ``share_memory_()``
        or normal pickling downstream).
    """
    if mm_inputs is None:
        return None
    return _wrap_recursive(mm_inputs, transport_mode, pool)


def _wrap_recursive(
    data: Any,
    transport_mode: TensorTransportMode,
    pool: Optional["MmItemMemoryPool"],
) -> Any:
    if isinstance(data, torch.Tensor) and data.is_cuda:
        return _wrap_cuda_tensor(data, transport_mode, pool)
    elif isinstance(data, dict):
        return {k: _wrap_recursive(v, transport_mode, pool) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        wrapped = [_wrap_recursive(item, transport_mode, pool) for item in data]
        return type(data)(wrapped)
    else:
        return data


def _wrap_cuda_tensor(
    tensor: torch.Tensor,
    transport_mode: TensorTransportMode,
    pool: Optional["MmItemMemoryPool"],
) -> Any:
    if transport_mode == "cuda_ipc":
        return TransportProxyTensor(tensor, transport_mode="cuda_ipc")

    if transport_mode == "cuda_ipc_pool":
        if pool is None:
            raise ValueError("pool must be provided for transport_mode='cuda_ipc_pool'")
        sync_meta, pool_slice = pool.get_slice_with_flag(tensor)
        if pool_slice is not None:
            # Copy tensor bytes into the pool slice
            pool_slice.copy_(tensor.view(torch.int8).view(-1), non_blocking=True)
            return CudaIpcTensorTransportProxy(
                data=pool_slice,
                info_data=tensor,
                sync_buffer_meta=sync_meta,
            )
        else:
            # Pool full – fall back to simple IPC (with potential memory leak)
            logger.warning(
                "Pool full; falling back to simple CUDA IPC (potential memory leak)"
            )
            return TransportProxyTensor(tensor, transport_mode="cuda_ipc")

    # "default" – move to CPU shared memory (handled by share_memory_() downstream)
    return tensor


def unwrap_mm_inputs_from_ipc(
    mm_inputs: Optional[Dict[str, Any]],
    device_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Recursively reconstruct tensors from IPC proxy objects.

    Call this in the *receiver* process after getting data from the queue.

    Args:
        mm_inputs: Data structure possibly containing IPC proxy objects.
        device_index: If not None, move reconstructed tensors to this device.
    """
    if mm_inputs is None:
        return None
    return _unwrap_recursive(mm_inputs, device_index)


def _unwrap_recursive(data: Any, device_index: Optional[int]) -> Any:
    if isinstance(data, CudaIpcTensorTransportProxy):
        return data.reconstruct_on_device(device_index)
    elif isinstance(data, TransportProxyTensor):
        # Already reconstructed during unpickling; just return as plain tensor
        return data.as_subclass(torch.Tensor)
    elif isinstance(data, dict):
        return {k: _unwrap_recursive(v, device_index) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        result = [_unwrap_recursive(item, device_index) for item in data]
        return type(data)(result)
    else:
        return data
