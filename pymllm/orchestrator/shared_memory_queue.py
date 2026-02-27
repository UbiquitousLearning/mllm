"""
Shared memory and queue utilities for fast IPC between tokenizer and scheduler.

This module implements shared-queue fast path to avoid expensive
ZMQ serialization of large multimodal tensors.

Design:
    - Metadata lane: Small tokenized objects stored in shared memory keyed by rid
    - Tensor lane: Large tensors made shareable via share_memory_() and passed by handle
"""

import logging
import pickle
import uuid
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class SharedMemoryManager:
    """Manages shared memory segments for passing metadata between processes.

    Each tokenized request's metadata is written to a unique shared memory segment
    keyed by its request ID (rid). The scheduler reads and immediately unlinks the
    segment to prevent memory leaks.
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
        # Serialize the metadata
        data = pickle.dumps(metadata)
        size = len(data)

        # Create unique shared memory segment name
        shm_name = f"pymllm_meta_{rid}_{uuid.uuid4().hex[:8]}"

        try:
            # Create shared memory segment
            shm = SharedMemory(name=shm_name, create=True, size=size)
            # Write data
            shm.buf[:size] = data
            shm.close()
            logger.debug(f"Wrote {size} bytes to shared memory {shm_name}")
            return shm_name
        except Exception as e:
            logger.error(f"Failed to write metadata to shared memory: {e}")
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
            # Open existing shared memory segment
            shm = SharedMemory(name=shm_name, create=False)
            # Read and deserialize data
            data = bytes(shm.buf[:])
            metadata = pickle.loads(data)
            shm.close()

            # Unlink to free memory immediately
            if unlink:
                try:
                    shm.unlink()
                    logger.debug(f"Read and unlinked shared memory {shm_name}")
                except FileNotFoundError:
                    # Already unlinked, ignore
                    pass

            return metadata
        except Exception as e:
            logger.error(f"Failed to read metadata from shared memory {shm_name}: {e}")
            raise

    @staticmethod
    def cleanup(shm_name: str) -> None:
        """Manually cleanup a shared memory segment (for error recovery)."""
        try:
            shm = SharedMemory(name=shm_name, create=False)
            shm.close()
            shm.unlink()
            logger.debug(f"Cleaned up shared memory {shm_name}")
        except FileNotFoundError:
            pass  # Already cleaned up
        except Exception as e:
            logger.warning(f"Failed to cleanup shared memory {shm_name}: {e}")


class TensorQueue:
    """Queue for passing large tensors between processes using shared memory.

    Tensors are made shareable via .share_memory_() and passed through a
    multiprocessing.Queue by handle (metadata only, not the actual data).
    """

    def __init__(self, maxsize: int = 0):
        """Initialize the tensor queue.

        Args:
            maxsize: Maximum queue size (0 for unlimited)
        """
        self._queue: Queue = Queue(maxsize=maxsize)

    def put(self, rid: str, shm_name: str, mm_inputs: Optional[Dict[str, Any]]) -> None:
        """Put a request with multimodal inputs into the queue.

        Args:
            rid: Request ID
            shm_name: Shared memory segment name for metadata
            mm_inputs: Multimodal inputs dict (can contain torch tensors)
        """
        # Make tensors shareable if present
        if mm_inputs is not None:
            mm_inputs = self._make_tensors_shareable(mm_inputs)

        self._queue.put((rid, shm_name, mm_inputs))
        logger.debug(f"Put request {rid} into tensor queue (shm={shm_name})")

    def get(
        self, timeout: Optional[float] = None
    ) -> tuple[str, str, Optional[Dict[str, Any]]]:
        """Get a request from the queue.

        Args:
            timeout: Timeout in seconds (None for blocking indefinitely)

        Returns:
            Tuple of (rid, shm_name, mm_inputs)
        """
        rid, shm_name, mm_inputs = self._queue.get(timeout=timeout)
        logger.debug(f"Got request {rid} from tensor queue (shm={shm_name})")
        return rid, shm_name, mm_inputs

    def empty(self) -> bool:
        """Check if the queue is empty."""
        return self._queue.empty()

    def qsize(self) -> int:
        """Return the approximate size of the queue."""
        try:
            return self._queue.qsize()
        except NotImplementedError:
            return 0  # Some platforms don't support qsize

    def close(self) -> None:
        """Close the queue."""
        self._queue.close()

    @staticmethod
    def _make_tensors_shareable(data: Any) -> Any:
        """Recursively make all torch tensors in a data structure shareable.

        Args:
            data: Nested dict/list/tensor structure

        Returns:
            The same structure with tensors made shareable via share_memory_()
        """
        if isinstance(data, torch.Tensor):
            # Make tensor shareable across processes
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
