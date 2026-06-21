"""Orchestrator module for distributed computation."""

from pymllm.orchestrator.group_coordinator import (
    GroupCoordinator,
    divide,
    split_tensor_along_dim,
)
from pymllm.orchestrator.parallel_state import (
    data_parallel_all_reduce,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_dp_group,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    initialize_model_parallel,
    model_parallel_is_initialized,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)

__all__ = [
    # GroupCoordinator
    "GroupCoordinator",
    "divide",
    "split_tensor_along_dim",
    # TP
    "get_tp_group",
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "tensor_model_parallel_all_reduce",
    "tensor_model_parallel_all_gather",
    # DP
    "get_dp_group",
    "get_data_parallel_rank",
    "get_data_parallel_world_size",
    "data_parallel_all_reduce",
    # PP
    "get_pp_group",
    "get_pipeline_model_parallel_rank",
    "get_pipeline_model_parallel_world_size",
    # State
    "initialize_model_parallel",
    "model_parallel_is_initialized",
]
