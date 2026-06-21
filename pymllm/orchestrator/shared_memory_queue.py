"""
Shared memory and queue utilities for fast IPC between tokenizer and scheduler.

This module implements the shared-queue fast path to avoid expensive ZMQ
serialization of large multimodal tensors.

## Design

- **Metadata lane**: Small tokenized objects are written to a POSIX shared memory
  segment keyed by the request ID (``rid``). The scheduler reads and immediately
  unlinks the segment.

- **Tensor lane**: Large tensors can be transported in one of three modes,
  controlled by ``TensorTransportMode`` (passed at queue construction time):

  * ``"default"``       – CPU tensors only. GPU tensors are moved to POSIX shared
    memory via ``tensor.share_memory_()`` (or left on CPU if already there).
    This is the original behaviour and requires no CUDA support.

  * ``"cuda_ipc"``      – GPU tensors stay on GPU and are wrapped in
    :class:`~pymllm.orchestrator.cuda_ipc_transport.TransportProxyTensor`. On the
    receiver side the proxy's ``__setstate__`` automatically reconstructs the
    tensor from the CUDA IPC handle during unpickling. CPU tensors are handled as
    in ``"default"`` mode. **Caveat**: GPU memory is not freed until the sender
    process exits (PyTorch limitation). Prefer ``"cuda_ipc_pool"`` for services.

  * ``"cuda_ipc_pool"`` – GPU tensors are copied into a pre-allocated
    :class:`~pymllm.orchestrator.cuda_ipc_transport.MmItemMemoryPool` workspace and
    wrapped in :class:`~pymllm.orchestrator.cuda_ipc_transport.CudaIpcTensorTransportProxy`.
    After the receiver copies the data it increments a sync flag and the sender's
    recycler thread returns the chunk to the pool. This avoids GPU memory leaks.
    CPU tensors are handled as in ``"default"`` mode.

## Key relationship with CUDA IPC

``"default"`` and ``"cuda_ipc*"`` modes are **mutually exclusive for GPU tensors**:

- In ``"default"`` mode, GPU tensors that need to cross process boundaries must
  first be moved to CPU (``share_memory_()``). This incurs a GPU→CPU copy.
- In ``"cuda_ipc*"`` modes, GPU tensors are shared as-is via CUDA IPC handles;
  no copy to CPU is needed.

CPU tensors are always handled via ``share_memory_()`` regardless of the mode.
"""

from __future__ import annotations

import logging
import pickle
import uuid
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Literal, Optional

import torch

from pymllm.orchestrator.cuda_ipc_transport import (
    MmItemMemoryPool,
    TensorTransportMode,
    unwrap_mm_inputs_from_ipc,
    wrap_mm_inputs_for_ipc,
)

logger = logging.getLogger(__name__)


class SharedMemoryManager:
    """Manages shared memory segments for passing metadata between processes.

    Each tokenized request's metadata is written to a unique shared memory
    segment keyed by its request ID (rid). The scheduler reads and immediately
    unlinks the segment to prevent memory leaks.
    """

    @staticmethod
    def write_metadata(rid: str, metadata: Any) -> str:
        """Write metadata to shared memory and return the segment name.

        Args:
            rid: Request ID (used as part of the shared memory name)
            metadata: Serializable metadata object

        Returns:
            str: The shared memory segment name
        """
        data = pickle.dumps(metadata)
        size = len(data)
        shm_name = f"pymllm_meta_{rid}_{uuid.uuid4().hex[:8]}"
        try:
            shm = SharedMemory(name=shm_name, create=True, size=size)
            shm.buf[:size] = data
            shm.close()
            logger.debug("Wrote %d bytes to shared memory %s", size, shm_name)
            return shm_name
        except Exception as exc:
            logger.error("Failed to write metadata to shared memory: %s", exc)
            raise

    @staticmethod
    def read_metadata(shm_name: str, unlink: bool = True) -> Any:
        """Read metadata from shared memory and optionally unlink it.

        Args:
            shm_name: The shared memory segment name
            unlink: If True, immediately unlink the segment after reading

        Returns:
            The deserialized metadata object
        """
        try:
            shm = SharedMemory(name=shm_name, create=False)
            data = bytes(shm.buf[:])
            metadata = pickle.loads(data)
            shm.close()
            if unlink:
                try:
                    shm.unlink()
                    logger.debug("Read and unlinked shared memory %s", shm_name)
                except FileNotFoundError:
                    pass
            return metadata
        except Exception as exc:
            logger.error(
                "Failed to read metadata from shared memory %s: %s", shm_name, exc
            )
            raise

    @staticmethod
    def cleanup(shm_name: str) -> None:
        """Manually cleanup a shared memory segment (for error recovery)."""
        try:
            shm = SharedMemory(name=shm_name, create=False)
            shm.close()
            shm.unlink()
            logger.debug("Cleaned up shared memory %s", shm_name)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to cleanup shared memory %s: %s", shm_name, exc)


class TensorQueue:
    """Queue for passing large tensors between processes.

    Depending on ``transport_mode``, GPU tensors are either moved to CPU shared
    memory (``"default"``) or kept on GPU and shared via CUDA IPC handles
    (``"cuda_ipc"`` / ``"cuda_ipc_pool"``).

    Args:
        maxsize: Maximum queue size (0 for unlimited).
        transport_mode: Controls how GPU tensors are transported.
        pool: Required when ``transport_mode == "cuda_ipc_pool"``.
    """

    def __init__(
        self,
        maxsize: int = 0,
        transport_mode: TensorTransportMode = "default",
        pool: Optional[MmItemMemoryPool] = None,
    ) -> None:
        # pool is allowed to be None at construction time for "cuda_ipc_pool" mode
        # because the pool is initialised lazily inside the sender subprocess.
        # The pool reference is injected later via _pool attribute assignment.
        self._queue: Queue = Queue(maxsize=maxsize)
        self._transport_mode = transport_mode
        self._pool = pool

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    def put(
        self,
        rid: str,
        shm_name: str,
        mm_inputs: Optional[Dict[str, Any]],
    ) -> None:
        """Put a request into the queue.

        GPU tensors inside *mm_inputs* are wrapped according to
        ``transport_mode`` before being placed into the underlying
        ``multiprocessing.Queue``.

        Args:
            rid: Request ID.
            shm_name: Shared memory segment name for the tokenized metadata.
            mm_inputs: Multimodal inputs dict (may contain CUDA tensors).
        """
        if mm_inputs is not None:
            if self._transport_mode in ("cuda_ipc", "cuda_ipc_pool"):
                if self._transport_mode == "cuda_ipc_pool" and self._pool is None:
                    # Pool not yet initialised (race condition or CUDA unavailable);
                    # fall back to simple CUDA IPC for this message.
                    effective_mode = "cuda_ipc"
                else:
                    effective_mode = self._transport_mode
                # Wrap CUDA tensors in IPC proxies (stays on GPU, no copy to CPU)
                mm_inputs = wrap_mm_inputs_for_ipc(
                    mm_inputs,
                    transport_mode=effective_mode,
                    pool=self._pool,
                )
                # CPU tensors within mm_inputs are still shared via share_memory_()
                mm_inputs = self._share_cpu_tensors(mm_inputs)
            else:
                # "default": move all tensors to CPU shared memory
                mm_inputs = self._make_tensors_shareable(mm_inputs)

        self._queue.put((rid, shm_name, mm_inputs))
        logger.debug("Put request %s into tensor queue (shm=%s)", rid, shm_name)

    # ------------------------------------------------------------------
    # Consumer side
    # ------------------------------------------------------------------

    def get(
        self, timeout: Optional[float] = None
    ) -> tuple[str, str, Optional[Dict[str, Any]]]:
        """Get a request from the queue.

        GPU tensors wrapped as IPC proxies are **not** automatically
        reconstructed here – the caller (scheduler) must call
        :func:`~pymllm.orchestrator.cuda_ipc_transport.unwrap_mm_inputs_from_ipc`
        after retrieval.

        Args:
            timeout: Timeout in seconds (None for blocking).

        Returns:
            Tuple of ``(rid, shm_name, mm_inputs)``.
        """
        rid, shm_name, mm_inputs = self._queue.get(timeout=timeout)
        logger.debug("Got request %s from tensor queue (shm=%s)", rid, shm_name)
        return rid, shm_name, mm_inputs

    # ------------------------------------------------------------------
    # Queue introspection
    # ------------------------------------------------------------------

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        try:
            return self._queue.qsize()
        except NotImplementedError:
            return 0

    def close(self) -> None:
        self._queue.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tensors_shareable(data: Any) -> Any:
        """Recursively move all tensors (CPU and CUDA) to POSIX shared memory.

        GPU tensors are first moved to CPU (incurring a device copy), then
        placed in shared memory.  This is the ``"default"`` path.
        """
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                data = data.cpu()
            if not data.is_shared():
                data = data.share_memory_()
            return data
        elif isinstance(data, dict):
            return {k: TensorQueue._make_tensors_shareable(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            result = [TensorQueue._make_tensors_shareable(item) for item in data]
            return type(data)(result)
        else:
            return data

    @staticmethod
    def _share_cpu_tensors(data: Any) -> Any:
        """Recursively place CPU tensors in shared memory (GPU tensors are already
        wrapped as IPC proxies and must not be touched here).
        """
        if isinstance(data, torch.Tensor) and not data.is_cuda:
            if not data.is_shared():
                data = data.share_memory_()
            return data
        elif isinstance(data, dict):
            return {k: TensorQueue._share_cpu_tensors(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            result = [TensorQueue._share_cpu_tensors(item) for item in data]
            return type(data)(result)
        else:
            return data
