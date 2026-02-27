"""Minimal parallel state for single-GPU serving.

pymllm targets single-GPU, high-concurrency inference. This module keeps
the TP / DP / PP scaffolding so the rest of the codebase can query ranks
and groups uniformly, but the default (and expected) case is world_size=1.
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist

from pymllm.orchestrator.group_coordinator import GroupCoordinator

logger = logging.getLogger(__name__)

_TP_GROUP: Optional[GroupCoordinator] = None
_DP_GROUP: Optional[GroupCoordinator] = None
_PP_GROUP: Optional[GroupCoordinator] = None

_TP_RANK: int = 0
_TP_SIZE: int = 1
_DP_RANK: int = 0
_DP_SIZE: int = 1
_PP_RANK: int = 0
_PP_SIZE: int = 1


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    data_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: str = "nccl",
) -> None:
    global _TP_GROUP, _DP_GROUP, _PP_GROUP
    global _TP_RANK, _TP_SIZE, _DP_RANK, _DP_SIZE, _PP_RANK, _PP_SIZE

    _TP_SIZE = tensor_model_parallel_size
    _DP_SIZE = data_parallel_size
    _PP_SIZE = pipeline_model_parallel_size

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    local_rank = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0

    assert (
        tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size
        == world_size
    ), (
        f"TP({tensor_model_parallel_size}) * DP({data_parallel_size}) * "
        f"PP({pipeline_model_parallel_size}) != World({world_size})"
    )

    logger.info(
        "Parallel init: world=%d rank=%d tp=%d dp=%d pp=%d",
        world_size,
        world_rank,
        tensor_model_parallel_size,
        data_parallel_size,
        pipeline_model_parallel_size,
    )

    if tensor_model_parallel_size > 1:
        num_tp_groups = world_size // tensor_model_parallel_size
        for i in range(num_tp_groups):
            ranks = list(
                range(
                    i * tensor_model_parallel_size,
                    (i + 1) * tensor_model_parallel_size,
                )
            )
            if world_rank in ranks:
                _TP_GROUP = GroupCoordinator(
                    ranks=ranks,
                    local_rank=local_rank,
                    backend=backend,
                )
                _TP_RANK = _TP_GROUP.rank_in_group
                break

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
                _DP_RANK = _DP_GROUP.rank_in_group
                break

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
                _PP_RANK = _PP_GROUP.rank_in_group
                break


# ---- group accessors ------------------------------------------------------


def get_tp_group() -> Optional[GroupCoordinator]:
    return _TP_GROUP


def get_dp_group() -> Optional[GroupCoordinator]:
    return _DP_GROUP


def get_pp_group() -> Optional[GroupCoordinator]:
    return _PP_GROUP


# ---- rank / size helpers --------------------------------------------------


def get_tensor_model_parallel_rank() -> int:
    return _TP_RANK


def get_tensor_model_parallel_world_size() -> int:
    return _TP_SIZE


def get_data_parallel_rank() -> int:
    return _DP_RANK


def get_data_parallel_world_size() -> int:
    return _DP_SIZE


def get_pipeline_model_parallel_rank() -> int:
    return _PP_RANK


def get_pipeline_model_parallel_world_size() -> int:
    return _PP_SIZE


def model_parallel_is_initialized() -> bool:
    return _TP_GROUP is not None or _DP_GROUP is not None or _PP_GROUP is not None


# ---- communication helpers ------------------------------------------------


def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    group = get_tp_group()
    if group is None:
        return tensor
    return group.all_reduce(tensor)


def tensor_model_parallel_all_gather(
    tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    group = get_tp_group()
    if group is None:
        return tensor
    return group.all_gather(tensor, dim=dim)


def data_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    group = get_dp_group()
    if group is None:
        return tensor
    return group.all_reduce(tensor)
