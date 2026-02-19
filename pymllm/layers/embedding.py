import torch
import torch.nn.functional as F
from torch.nn import Parameter

from pymllm.layers.base import MllmBaseLayer
from pymllm.layers.utils import set_weight_attrs
from pymllm.orchestrator import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)


class VocabParallelEmbedding(MllmBaseLayer):
    """Embedding layer with vocabulary parallelism.

    This layer shards the embedding table along the vocabulary dimension
    for tensor parallelism.

    Args:
        num_embeddings: Size of the vocabulary.
        embedding_dim: Size of the embedding vector.
        padding_idx: Index for padding token (optional).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
    ):
        super().__init__()

        # Get TP info from global state
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Calculate sharded size
        if self.num_embeddings % self.tp_size != 0:
            raise ValueError(
                f"num_embeddings ({num_embeddings}) must be divisible by "
                f"tp_size ({self.tp_size})"
            )

        self.num_embeddings_per_partition = divide(num_embeddings, self.tp_size)

        # Create sharded weight
        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )

        # Calculate shard range
        self.vocab_start_index = self.tp_rank * self.num_embeddings_per_partition
        self.vocab_end_index = (
            self.vocab_start_index + self.num_embeddings_per_partition
        )

        # Set weight attributes for loading
        set_weight_attrs(
            self.weight,
            {
                "output_dim": 0,  # Shard along vocab dimension
                "input_dim": 1,  # Embedding dimension
                "weight_loader": self.weight_loader,
            },
        )

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """Load sharded weights into the parameter.

        Args:
            param: The parameter to load weights into.
            loaded_weight: The weight tensor loaded from checkpoint (full size).
        """
        output_dim = getattr(param, "output_dim", None)

        if output_dim is None or self.tp_size == 1:
            # No sharding, direct copy
            assert param.data.shape == loaded_weight.shape, (
                f"Shape mismatch: param {param.data.shape} vs "
                f"loaded {loaded_weight.shape}"
            )
            param.data.copy_(loaded_weight)
        else:
            # Sharded loading: slice the loaded weight
            assert loaded_weight.shape[output_dim] == self.num_embeddings, (
                f"Loaded weight vocab size {loaded_weight.shape[output_dim]} "
                f"does not match expected {self.num_embeddings}"
            )

            # Slice along vocab dimension
            if output_dim == 0:
                shard_weight = loaded_weight[
                    self.vocab_start_index : self.vocab_end_index, :
                ]
            else:
                shard_weight = loaded_weight.narrow(
                    output_dim,
                    self.vocab_start_index,
                    self.num_embeddings_per_partition,
                )

            assert param.data.shape == shard_weight.shape, (
                f"Shard shape mismatch: param {param.data.shape} vs "
                f"shard {shard_weight.shape}"
            )
            param.data.copy_(shard_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the embedding layer with TP support.

        Args:
            x: Input tensor of token ids.

        Returns:
            Embedded representation (all-reduced across TP group if needed).
        """
        local_padding_idx = self.padding_idx
        if self.tp_size > 1:
            # Create mask for valid vocab range
            vocab_mask = (x >= self.vocab_start_index) & (x < self.vocab_end_index)

            # Adjust indices to local vocab space
            masked_input = torch.where(
                vocab_mask,
                x - self.vocab_start_index,
                torch.zeros_like(x),  # Invalid indices become 0 (will be masked)
            )
            # F.embedding expects indices in local weight-table space.
            # Only pass padding_idx on the owning rank, remapped to local offset.
            if self.padding_idx is not None:
                if self.vocab_start_index <= self.padding_idx < self.vocab_end_index:
                    local_padding_idx = self.padding_idx - self.vocab_start_index
                else:
                    local_padding_idx = None
        else:
            masked_input = x
            vocab_mask = None

        # Lookup embeddings
        output = F.embedding(
            masked_input.long(),
            self.weight,
            padding_idx=local_padding_idx,
        )

        # Mask invalid positions (for TP)
        if vocab_mask is not None:
            output.masked_fill_(~vocab_mask.unsqueeze(-1), 0)

        # All-reduce across TP group
        if self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)

        return output
