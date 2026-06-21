"""GroupCoordinator for distributed communication."""

from typing import List
import torch
import torch.distributed as dist


class GroupCoordinator:
    """Manages a group of processes for distributed communication.
    
    Lightweight wrapper around torch.distributed.ProcessGroup.
    
    Args:
        ranks: List of global ranks in this group
        local_rank: Local rank for device assignment
        backend: Backend to use (nccl, gloo, etc.)
    """
    
    def __init__(
        self,
        ranks: List[int],
        local_rank: int,
        backend: str = "nccl",
    ):
        self.ranks = ranks
        self.local_rank = local_rank
        self.backend = backend
        self.world_size = len(ranks)
        
        # Get rank in this specific group
        self.rank_in_group = ranks.index(dist.get_rank()) if dist.is_initialized() else 0
        
        # Create process group
        if dist.is_initialized() and self.world_size > 1:
            self.device_group = dist.new_group(ranks, backend=backend)
        else:
            self.device_group = None
    
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce across the group."""
        if self.device_group is not None:
            dist.all_reduce(tensor, group=self.device_group)
        return tensor
    
    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """All-gather across the group."""
        if self.device_group is None:
            return tensor
        
        world_size = self.world_size
        if dim == 0:
            shape = list(tensor.shape)
            shape[0] = shape[0] * world_size
            output = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
            dist.all_gather_into_tensor(output, tensor, group=self.device_group)
            return output
        else:
            # For non-dim-0 gathers, use tensor list
            tensor_list = [
                torch.empty_like(tensor) for _ in range(world_size)
            ]
            dist.all_gather(tensor_list, tensor, group=self.device_group)
            return torch.cat(tensor_list, dim=dim)
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast from source rank to all.

        Args:
            tensor: Tensor to broadcast.
            src: Source rank relative to this group (0 <= src < world_size).
        """
        if self.device_group is not None:
            global_src = self.ranks[src]
            dist.broadcast(tensor, src=global_src, group=self.device_group)
        return tensor


def divide(numerator: int, denominator: int) -> int:
    """Divide and ensure divisibility."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )
    return numerator // denominator


def split_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    """Split tensor along a dimension for tensor parallelism."""
    dim_size = tensor.size(dim)
    assert dim_size % world_size == 0, (
        f"Dimension {dim} ({dim_size}) not divisible by world_size {world_size}"
    )
    
    chunk_size = dim_size // world_size
    start = rank * chunk_size
    end = start + chunk_size
    
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, end)
    return tensor[tuple(slices)]
