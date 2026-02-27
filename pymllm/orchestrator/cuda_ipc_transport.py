"""
CUDA IPC Transport for zero-copy tensor sharing between processes.

This module implements CUDA IPC with workspace buffer management
to avoid PyTorch's memory leak issue when sharing IPC handles.

1. Create a workspace buffer on GPU (pre-allocated memory pool)
2. Copy tensor data to a chunk in the workspace
3. Get CUDA IPC handle for the chunk
4. Send handle + metadata (shape, dtype, offset) to another process
5. Reconstruct tensor in target process from IPC handle
6. Copy to local tensor and mark chunk as reusable

Key Problem Solved:
    PyTorch never releases tensors whose IPC handles are shared until process ends.
    Solution: Use a fixed-size workspace buffer and recycle chunks.
"""

import logging
import struct
import uuid
from dataclasses import dataclass
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.cuda as cuda

logger = logging.getLogger(__name__)


@dataclass
class MemoryChunk:
    """Represents a chunk in the workspace buffer."""

    offset: int  # Offset in bytes from workspace start
    size: int  # Size in bytes
    in_use: bool  # Whether the chunk is currently occupied
    sync_shm_name: Optional[str] = None  # Shared memory name for sync flag


class WorkspaceBuffer:
    """GPU memory pool for storing multimodal tensors temporarily.

    This prevents the PyTorch IPC handle memory leak by using a fixed-size
    pre-allocated buffer and recycling chunks.
    """

    def __init__(self, size_gb: float = 4.0, device: int = 0):
        """Initialize workspace buffer.

        Args:
            size_gb: Total size of workspace in GB
            device: CUDA device ID
        """
        self.device = device
        self.total_size = int(size_gb * 1024 * 1024 * 1024)  # Convert GB to bytes

        # Allocate workspace on GPU
        with torch.cuda.device(device):
            self.workspace = torch.empty(
                self.total_size // 4,  # Divide by 4 because we use float32
                dtype=torch.float32,
                device=f"cuda:{device}",
            )

        # Initialize chunk management
        self.chunks: List[MemoryChunk] = [
            MemoryChunk(offset=0, size=self.total_size, in_use=False)
        ]

        # Container for reusable sync buffers
        self.sync_buffer_pool: List[str] = []

        logger.info(
            f"WorkspaceBuffer initialized: {size_gb}GB on cuda:{device}, "
            f"ptr={self.workspace.data_ptr():#x}"
        )

    def allocate(self, size_bytes: int) -> Optional[Tuple[int, str]]:
        """Allocate a chunk from the workspace.

        Args:
            size_bytes: Required size in bytes

        Returns:
            Tuple of (offset, sync_shm_name) if successful, None if no space
        """
        # Find a free chunk that's large enough
        for i, chunk in enumerate(self.chunks):
            if not chunk.in_use and chunk.size >= size_bytes:
                # Mark chunk as in use
                chunk.in_use = True

                # Get or create sync buffer
                if self.sync_buffer_pool:
                    sync_shm_name = self.sync_buffer_pool.pop()
                    # Reset sync flag to 0 (not ready)
                    self._reset_sync_buffer(sync_shm_name)
                else:
                    sync_shm_name = self._create_sync_buffer()

                chunk.sync_shm_name = sync_shm_name

                # If chunk is larger than needed, split it
                if chunk.size > size_bytes:
                    # Create a new free chunk for the remaining space
                    new_chunk = MemoryChunk(
                        offset=chunk.offset + size_bytes,
                        size=chunk.size - size_bytes,
                        in_use=False,
                    )
                    chunk.size = size_bytes
                    self.chunks.insert(i + 1, new_chunk)

                logger.debug(
                    f"Allocated chunk: offset={chunk.offset}, size={size_bytes}, "
                    f"sync_shm={sync_shm_name}"
                )
                return chunk.offset, sync_shm_name

        logger.warning(f"WorkspaceBuffer: No space for {size_bytes} bytes")
        return None

    def release(self, offset: int) -> None:
        """Release a chunk back to the pool.

        Args:
            offset: Offset of the chunk to release
        """
        for i, chunk in enumerate(self.chunks):
            if chunk.offset == offset and chunk.in_use:
                chunk.in_use = False

                # Return sync buffer to pool
                if chunk.sync_shm_name:
                    self.sync_buffer_pool.append(chunk.sync_shm_name)
                    chunk.sync_shm_name = None

                # Try to merge with adjacent free chunks
                self._merge_chunks()

                logger.debug(f"Released chunk: offset={offset}")
                return

        logger.warning(f"Attempted to release unknown chunk at offset {offset}")

    def _merge_chunks(self) -> None:
        """Merge adjacent free chunks to reduce fragmentation."""
        i = 0
        while i < len(self.chunks) - 1:
            current = self.chunks[i]
            next_chunk = self.chunks[i + 1]

            if not current.in_use and not next_chunk.in_use:
                # Merge chunks
                current.size += next_chunk.size

                # Keep first chunk's sync buffer, return second to pool
                if next_chunk.sync_shm_name:
                    self.sync_buffer_pool.append(next_chunk.sync_shm_name)

                self.chunks.pop(i + 1)
            else:
                i += 1

    def _create_sync_buffer(self) -> str:
        """Create a new shared memory sync buffer (8 bytes, initialized to 0)."""
        shm_name = f"pymllm_sync_{uuid.uuid4().hex[:12]}"
        shm = SharedMemory(name=shm_name, create=True, size=8)
        # Initialize to 0 (not ready)
        shm.buf[:8] = struct.pack("Q", 0)
        shm.close()
        logger.debug(f"Created sync buffer: {shm_name}")
        return shm_name

    def _reset_sync_buffer(self, shm_name: str) -> None:
        """Reset sync buffer to 0 (not ready)."""
        try:
            shm = SharedMemory(name=shm_name, create=False)
            shm.buf[:8] = struct.pack("Q", 0)
            shm.close()
        except Exception as e:
            logger.warning(f"Failed to reset sync buffer {shm_name}: {e}")

    def copy_tensor_to_workspace(self, tensor: torch.Tensor, offset: int) -> None:
        """Copy tensor data to workspace at given offset.

        Args:
            tensor: Source tensor (must be on same CUDA device)
            offset: Byte offset in workspace
        """
        if not tensor.is_cuda or tensor.device.index != self.device:
            raise ValueError(f"Tensor must be on cuda:{self.device}")

        size_bytes = tensor.numel() * tensor.element_size()

        # Get view of workspace at offset
        offset_elements = offset // 4  # Workspace is float32
        num_elements = (size_bytes + 3) // 4  # Round up

        workspace_view = self.workspace[
            offset_elements : offset_elements + num_elements
        ]

        # Copy tensor data (flatten and cast to float32 view)
        tensor_flat = tensor.flatten().view(torch.uint8)
        workspace_flat = workspace_view.view(torch.uint8)[: tensor_flat.numel()]
        workspace_flat.copy_(tensor_flat)

        logger.debug(f"Copied tensor {tensor.shape} to workspace offset {offset}")

    def get_ipc_handle(self) -> bytes:
        """Get CUDA IPC handle for the workspace buffer.

        Returns:
            CUDA IPC handle as bytes
        """
        # Get IPC handle using torch.cuda API
        # Note: This requires CUDA-capable device with IPC support
        handle = cuda.cudart().cudaIpcGetMemHandle(self.workspace.data_ptr())
        return bytes(handle)

    def cleanup(self) -> None:
        """Cleanup all sync buffers."""
        all_shm_names = set()
        for chunk in self.chunks:
            if chunk.sync_shm_name:
                all_shm_names.add(chunk.sync_shm_name)
        all_shm_names.update(self.sync_buffer_pool)

        for shm_name in all_shm_names:
            try:
                shm = SharedMemory(name=shm_name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.warning(f"Failed to cleanup sync buffer {shm_name}: {e}")

        logger.info("WorkspaceBuffer cleaned up")


@dataclass
class TensorMetadata:
    """Metadata for reconstructing a tensor from CUDA IPC handle."""

    shape: Tuple[int, ...]
    dtype: torch.dtype
    offset: int  # Byte offset in workspace
    size_bytes: int
    sync_shm_name: str  # Shared memory name for sync flag


class CudaIPCTransport:
    """Transport for sharing CUDA tensors via IPC handles."""

    def __init__(
        self,
        workspace_size_gb: float = 4.0,
        device: int = 0,
    ):
        """Initialize CUDA IPC transport.

        Args:
            workspace_size_gb: Size of workspace buffer in GB
            device: CUDA device ID
        """
        self.device = device
        self.workspace = WorkspaceBuffer(workspace_size_gb, device)
        self.ipc_handle = self.workspace.get_ipc_handle()
        self.queue: Queue = Queue()

    def send_tensor(self, rid: str, tensor: torch.Tensor) -> bool:
        """Send a tensor via CUDA IPC.

        Args:
            rid: Request ID
            tensor: Tensor to send (must be on CUDA)

        Returns:
            True if sent via CUDA IPC, False if fallback needed
        """
        if not tensor.is_cuda:
            logger.debug(f"Tensor for {rid} not on CUDA, skipping IPC")
            return False

        size_bytes = tensor.numel() * tensor.element_size()

        # Try to allocate from workspace
        result = self.workspace.allocate(size_bytes)
        if result is None:
            logger.warning(
                f"WorkspaceBuffer full, falling back to shared queue for {rid}"
            )
            return False

        offset, sync_shm_name = result

        # Copy tensor to workspace
        self.workspace.copy_tensor_to_workspace(tensor, offset)

        # Create metadata
        metadata = TensorMetadata(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            offset=offset,
            size_bytes=size_bytes,
            sync_shm_name=sync_shm_name,
        )

        # Send metadata through queue
        self.queue.put((rid, metadata, self.ipc_handle))

        logger.debug(f"Sent tensor {tensor.shape} for {rid} via CUDA IPC")
        return True

    def receive_tensor(
        self, timeout: float = 0.0001
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Receive a tensor via CUDA IPC.

        Args:
            timeout: Timeout for queue.get

        Returns:
            Tuple of (rid, tensor) or None if queue empty
        """
        try:
            rid, metadata, ipc_handle = self.queue.get(timeout=timeout)
        except Exception:
            return None

        # Open IPC memory handle
        # Note: This creates a tensor view into the remote process's workspace
        with torch.cuda.device(self.device):
            # Reconstruct tensor from IPC handle
            # This is a view into remote memory, we need to copy it locally

            # For now, use a simpler approach: signal to copy later
            # In production, you'd use cuda.cudart().cudaIpcOpenMemHandle

            logger.warning(
                "CUDA IPC receive not fully implemented - requires cudaIpcOpenMemHandle"
            )
            # TODO: Implement actual IPC handle opening

            # Create local tensor and signal copy completion
            tensor = torch.empty(
                metadata.shape, dtype=metadata.dtype, device=f"cuda:{self.device}"
            )

            # Mark chunk as ready for reuse by setting sync flag
            self._mark_chunk_reusable(metadata.sync_shm_name)

            return rid, tensor

    def _mark_chunk_reusable(self, sync_shm_name: str) -> None:
        """Mark a chunk as reusable by setting sync flag to 1."""
        try:
            shm = SharedMemory(name=sync_shm_name, create=False)
            shm.buf[:8] = struct.pack("Q", 1)  # Set to 1 (ready for reuse)
            shm.close()
            logger.debug(f"Marked chunk reusable: {sync_shm_name}")
        except Exception as e:
            logger.error(f"Failed to mark chunk reusable {sync_shm_name}: {e}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.workspace.cleanup()
        self.queue.close()
