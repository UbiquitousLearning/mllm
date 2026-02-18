"""Parallel state management for tensor and pipeline parallelism."""

import logging
import torch
import torch.distributed as dist
from typing import Optional

from pymllm.configs.global_config import get_global_config
from pymllm.orchestrator.group_coordinator import GroupCoordinator

logger = logging.getLogger(__name__)


# Global groups
_TP_GROUP: Optional[GroupCoordinator] = None
_DP_GROUP: Optional[GroupCoordinator] = None
_PP_GROUP: Optional[GroupCoordinator] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    data_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: str = "nccl",
) -> None:
    """Initialize model parallel groups.

    Args:
        tensor_model_parallel_size: Number of GPUs for tensor parallelism
        data_parallel_size: Number of GPUs for data parallelism
        pipeline_model_parallel_size: Number of stages for pipeline parallelism
        backend: Communication backend (nccl for GPU, gloo for CPU)
    """
    global _TP_GROUP, _DP_GROUP, _PP_GROUP

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0

    config = get_global_config()

    # Update runtime config
    config.runtime.world_size = world_size
    config.runtime.world_rank = world_rank
    config.runtime.local_rank = local_rank
    config.runtime.tp_size = tensor_model_parallel_size
    config.runtime.dp_size = data_parallel_size
    config.runtime.pp_size = pipeline_model_parallel_size

    # Logging
    logger.info(
        "Model parallel runtime config set: world_size=%s, world_rank=%s, "
        "local_rank=%s, tp_size=%s, dp_size=%s, pp_size=%s",
        config.runtime.world_size,
        config.runtime.world_rank,
        config.runtime.local_rank,
        config.runtime.tp_size,
        config.runtime.dp_size,
        config.runtime.pp_size,
    )

    # Validate parallelism setup
    assert (
        tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size
        == world_size
    ), (
        f"TP({tensor_model_parallel_size}) * DP({data_parallel_size}) * "
        f"PP({pipeline_model_parallel_size}) != World({world_size})"
    )

    # Create TP groups (intra-layer sharding)
    if tensor_model_parallel_size > 1:
        num_tp_groups = world_size // tensor_model_parallel_size
        for i in range(num_tp_groups):
            ranks = list(
                range(
                    i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size
                )
            )
            if world_rank in ranks:
                _TP_GROUP = GroupCoordinator(
                    ranks=ranks,
                    local_rank=local_rank,
                    backend=backend,
                )
                config.runtime.tp_rank = _TP_GROUP.rank_in_group
                break
    else:
        _TP_GROUP = None
        config.runtime.tp_rank = 0

    # Create DP groups (data replication)
    if data_parallel_size > 1:
        num_dp_groups = world_size // data_parallel_size
        for i in range(num_dp_groups):
            ranks = list(range(i, world_size, num_dp_groups))
            if world_rank in ranks:
                _DP_GROUP = GroupCoordinator(
                    ranks=ranks,
                    local_rank=local_rank,
                    backend=backend,
                )
                config.runtime.dp_rank = _DP_GROUP.rank_in_group
                break
    else:
        _DP_GROUP = None
        config.runtime.dp_rank = 0

    # Create PP groups (inter-layer partitioning)
    if pipeline_model_parallel_size > 1:
        num_pp_groups = world_size // pipeline_model_parallel_size
        for i in range(num_pp_groups):
            start = i * pipeline_model_parallel_size
            ranks = list(range(start, start + pipeline_model_parallel_size))
            if world_rank in ranks:
                _PP_GROUP = GroupCoordinator(
                    ranks=ranks,
                    local_rank=local_rank,
                    backend=backend,
                )
                config.runtime.pp_rank = _PP_GROUP.rank_in_group
                break
    else:
        _PP_GROUP = None
        config.runtime.pp_rank = 0


def get_tp_group() -> Optional[GroupCoordinator]:
    """Get the tensor model parallel group."""
    return _TP_GROUP


def get_dp_group() -> Optional[GroupCoordinator]:
    """Get the data parallel group."""
    return _DP_GROUP


def get_pp_group() -> Optional[GroupCoordinator]:
    """Get the pipeline parallel group."""
    return _PP_GROUP


# Convenience functions for tensor parallelism
def get_tensor_model_parallel_rank() -> int:
    """Get current tensor model parallel rank."""
    return get_global_config().runtime.tp_rank


def get_tensor_model_parallel_world_size() -> int:
    """Get tensor model parallel world size."""
    return get_global_config().runtime.tp_size


def get_data_parallel_rank() -> int:
    """Get current data parallel rank."""
    return get_global_config().runtime.dp_rank


def get_data_parallel_world_size() -> int:
    """Get data parallel world size."""
    return get_global_config().runtime.dp_size


def get_pipeline_model_parallel_rank() -> int:
    """Get current pipeline parallel rank."""
    return get_global_config().runtime.pp_rank


def get_pipeline_model_parallel_world_size() -> int:
    """Get pipeline parallel world size."""
    return get_global_config().runtime.pp_size


def model_parallel_is_initialized() -> bool:
    """Check if model parallel is initialized."""
    return _TP_GROUP is not None or _DP_GROUP is not None or _PP_GROUP is not None


# Communication helpers
def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce across TP group."""
    group = get_tp_group()
    if group is None:
        return tensor
    return group.all_reduce(tensor)


def tensor_model_parallel_all_gather(
    tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """All-gather across TP group."""
    group = get_tp_group()
    if group is None:
        return tensor
    return group.all_gather(tensor, dim=dim)


def data_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce across DP group."""
    group = get_dp_group()
    if group is None:
        return tensor
    return group.all_reduce(tensor)
