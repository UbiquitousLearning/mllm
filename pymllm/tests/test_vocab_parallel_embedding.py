"""Tests for VocabParallelEmbedding layer.

This module tests the VocabParallelEmbedding layer with and without
tensor parallelism.
"""

import os
import logging
import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from typing import Callable

from pymllm.layers import VocabParallelEmbedding
from pymllm.orchestrator import initialize_model_parallel
from pymllm.orchestrator.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

# Show runtime init logs during test execution.
logging.basicConfig(level=logging.INFO, force=True)
logging.getLogger().setLevel(logging.INFO)


# =============================================================================
# Helper: weight loading
# =============================================================================
def load_weight(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Load weight using the weight_loader attached to param attribute."""
    weight_loader = getattr(param, "weight_loader", None)
    if weight_loader is None:
        # Fallback: direct copy
        param.data.copy_(loaded_weight)
    else:
        # Call the loader attached to param
        weight_loader(param, loaded_weight)


# =============================================================================
# Real distributed tests with world_size=8 on CUDA
# =============================================================================
def run_worker_tp8_cuda(
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    test_func: Callable,
    return_dict: dict,
):
    """Worker function for multi-process testing with TP=8 on CUDA.

    Args:
        rank: Global rank across all nodes
        local_rank: Local rank within this node (used for GPU binding)
        world_size: Total number of processes across all nodes
        local_world_size: Number of processes on this node
        test_func: Test function to run
        return_dict: Shared dict for returning results
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Set device using local_rank (binds to GPU 0,1,2,3 on this node)
    torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    initialize_model_parallel(tensor_model_parallel_size=8)

    try:
        result = test_func(rank, local_rank, world_size)
        return_dict[rank] = result
    except Exception as e:
        import traceback

        return_dict[rank] = f"ERROR: {e}\n{traceback.format_exc()}"
    finally:
        torch.distributed.destroy_process_group()


def embedding_forward_tp8_worker_cuda(rank: int, local_rank: int, world_size: int):
    """Test forward pass with real TP=8 on CUDA.

    Args:
        rank: Global rank
        local_rank: Local rank within this node (for logging/debugging)
        world_size: Total world size
    """
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    assert tp_size == 8, f"Rank {rank}: tp_size should be 8"
    assert tp_rank == rank, f"Rank {rank}: tp_rank mismatch"

    vocab_size = 1024
    embed_dim = 64
    # .cuda() uses the device set by torch.cuda.set_device(local_rank)
    layer = VocabParallelEmbedding(vocab_size, embed_dim).cuda()

    # Verify the layer is on the correct GPU
    assert layer.weight.device.index == local_rank, (
        f"Rank {rank}: weight should be on GPU {local_rank}, got {layer.weight.device}"
    )

    expected_shard_size = vocab_size // 8
    assert layer.num_embeddings_per_partition == expected_shard_size
    assert layer.weight.shape == (expected_shard_size, embed_dim)

    # Each rank initializes its own shard with known pattern
    with torch.no_grad():
        layer.weight.fill_(float(rank + 1))  # Rank 0: 1.0, Rank 1: 2.0, ...

    # Create input on the correct GPU
    input_ids = torch.tensor([[0, 128, 256, 384], [512, 640, 768, 896]], device="cuda")

    output = layer(input_ids)
    assert output.shape == (2, 4, embed_dim)

    # Verify output is on correct GPU
    assert output.device.index == local_rank, (
        f"Rank {rank}: output should be on GPU {local_rank}, got {output.device}"
    )

    if rank == 0:
        # Each token is owned by exactly one TP rank. Since each rank fills its
        # local shard with (rank + 1), post-all-reduce output must match below.
        expected_token_values = torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            device=output.device,
            dtype=output.dtype,
        )
        expected_output = expected_token_values.unsqueeze(-1).expand(-1, -1, embed_dim)

        if torch.equal(output, expected_output):
            return "PASSED"
        return "FAILED: embedding output does not match expected TP aggregation"

    return "OK"


def weight_loading_tp8_worker_cuda(rank: int, local_rank: int, world_size: int):
    """Test weight loading with real TP=8 on CUDA.

    Args:
        rank: Global rank
        local_rank: Local rank within this node (for GPU binding verification)
        world_size: Total world size
    """
    vocab_size = 1024
    embed_dim = 64
    layer = VocabParallelEmbedding(vocab_size, embed_dim).cuda()

    # Verify the layer is on the correct GPU
    assert layer.weight.device.index == local_rank, (
        f"Rank {rank}: weight should be on GPU {local_rank}, got {layer.weight.device}"
    )

    full_weight = torch.randn(vocab_size, embed_dim)
    load_weight(layer.weight, full_weight.cuda())

    shard_size = vocab_size // 8
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size
    expected_shard = full_weight[start_idx:end_idx]

    if not torch.allclose(layer.weight.cpu(), expected_shard):
        return f"FAILED: shard mismatch at rank {rank}"

    if rank == 0:
        gathered_shards = [layer.weight.cpu().clone()]
        for other_rank in range(1, 8):
            other_shard = full_weight[
                other_rank * shard_size : (other_rank + 1) * shard_size
            ]
            gathered_shards.append(other_shard)

        reconstructed = torch.cat(gathered_shards, dim=0)
        if torch.allclose(reconstructed, full_weight):
            return "PASSED"
        else:
            return "FAILED: reconstruction mismatch"

    return "OK"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires at least 8 GPUs")
class TestVocabParallelEmbeddingRealTP8:
    """Real distributed tests with world_size=8 and TP=8 on CUDA."""

    def test_forward_pass_tp8_real(self):
        """Test forward pass with real TP=8 using 8 processes on CUDA."""
        world_size = 8
        local_world_size = 8  # Single node with 8 GPUs

        mp.set_start_method("spawn", force=True)

        manager = mp.Manager()
        return_dict = manager.dict()

        processes = []
        for rank in range(world_size):
            # In single-node setup, local_rank == rank
            local_rank = rank
            p = mp.Process(
                target=run_worker_tp8_cuda,
                args=(
                    rank,
                    local_rank,
                    world_size,
                    local_world_size,
                    embedding_forward_tp8_worker_cuda,
                    return_dict,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
                p.join()

        for rank in range(world_size):
            result = return_dict.get(rank, "TIMEOUT")
            if rank == 0:
                assert result == "PASSED", f"Rank {rank} failed: {result}"
            else:
                assert "ERROR" not in str(result), f"Rank {rank} error: {result}"

    def test_weight_loading_tp8_real(self):
        """Test weight loading with real TP=8 using 8 processes on CUDA."""
        world_size = 8
        local_world_size = 8  # Single node with 8 GPUs

        mp.set_start_method("spawn", force=True)

        manager = mp.Manager()
        return_dict = manager.dict()

        processes = []
        for rank in range(world_size):
            # In single-node setup, local_rank == rank
            local_rank = rank
            p = mp.Process(
                target=run_worker_tp8_cuda,
                args=(
                    rank,
                    local_rank,
                    world_size,
                    local_world_size,
                    weight_loading_tp8_worker_cuda,
                    return_dict,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
                p.join()

        for rank in range(world_size):
            result = return_dict.get(rank, "TIMEOUT")
            if rank == 0:
                assert result == "PASSED", f"Rank {rank} failed: {result}"
            else:
                assert "ERROR" not in str(result), f"Rank {rank} error: {result}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestVocabParallelEmbeddingCUDA:
    """Tests for non-parallel TP=1 mode on CUDA."""

    @pytest.fixture(autouse=True)
    def setup_config(self):
        import pymllm.orchestrator.parallel_state as ps
        ps._TP_SIZE = 1
        ps._TP_RANK = 0
        yield
        ps._TP_SIZE = 1
        ps._TP_RANK = 0

    def test_cuda_forward(self):
        layer = VocabParallelEmbedding(1000, 512).cuda()
        input_ids = torch.randint(0, 1000, (4, 32), device="cuda")

        output = layer(input_ids)

        assert output.device.type == "cuda"
        assert output.shape == (4, 32, 512)

    def test_cuda_weight_loader(self):
        layer = VocabParallelEmbedding(100, 64).cuda()

        cpu_weight = torch.randn(100, 64)
        load_weight(layer.weight, cpu_weight.cuda())

        assert torch.allclose(layer.weight.cpu(), cpu_weight)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
