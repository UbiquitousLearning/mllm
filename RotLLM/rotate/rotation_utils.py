from typing import Iterable
import torch
from torch import nn

from rotate.common import NormLinearIterator
from .hadamard_utils import random_hadamard_matrix

def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    
    Args:
    size (int): The size of the matrix (size x size).
    
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, device="cpu"):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')


class RMSNorm(nn.Module):
    def __init__(self, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"eps={self.variance_epsilon}"
    

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    norm_weight = layernorm.weight.data.double()
    norm_bias = layernorm.bias.data.double() if hasattr(layernorm, 'bias') else None
    norm_dim = norm_weight.shape[0]
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        linear_device = linear.weight.device
        in_dim = linear.in_features
        
        # this is for ViT merger
        # merger takes in merge_size * merge_size patches
        # while norm takes in only one patch
        repeat_times = in_dim // norm_dim
        
        if in_dim % norm_dim != 0:
            raise ValueError(f"Linear layer {linear} has in_features {in_dim} not divisible by LayerNorm {layernorm} with weight {norm_weight.shape[0]}")

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * (norm_weight.to(linear_device).repeat(repeat_times))).to(linear_dtype)

        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, norm_bias.to(linear_device).repeat(repeat_times))
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model: nn.Module, replace_ln: bool = False, verbose=False) -> None:
    it = NormLinearIterator.from_model(model)

    for father, norm_name, linears in it:
        # fuse the linear operations in Layernorm into the adjacent linear blocks.
        norm = getattr(father, norm_name)
        if verbose:
            print(f"Fusing {norm_name}")
            print(f"  {norm}")
            print(f"  {linears}")
            
        fuse_ln_linear(norm, linears)
        if not replace_ln: # keep the original layernorm/RMSNorm
            W_norm = norm.weight.data
            norm.weight.data = torch.ones_like(W_norm)
            if hasattr(norm, 'bias'):
                b_norm = norm.bias.data
                norm.bias.data = torch.zeros_like(b_norm)
        else:
            # eps = 1e-6
            # if hasattr(norm, 'variance_epsilon'):
            #     eps = norm.variance_epsilon
            # if hasattr(norm, 'eps'):
            #     eps = norm.eps
            # # replace the layernorm with RMSNorm
            # new_norm = RMSNorm(eps=eps)
            # setattr(father, norm_name, new_norm)
            
            from torch.nn import RMSNorm
            
            # in this case, we replace the layernorm with RMSNorm implemented by torch
            # torch's RMSNorm has weight
            # in some cases, we need to save the weight though it is always 1.0
            # our implementation of RMSNorm does not have weight
            # so here we use torch's RMSNorm
            # and set the weight to 1.0
            eps = getattr(norm, 'eps', getattr(norm, 'variance_epsilon', 1e-6))
            normalized_shape = norm.normalized_shape if hasattr(norm, 'normalized_shape') else norm.weight.shape
        
            device = norm.weight.device
            dtype = norm.weight.dtype
            new_norm = RMSNorm(normalized_shape=normalized_shape, eps=eps).to(device=device, dtype=dtype)
            new_norm.weight.data.fill_(1.0)
            setattr(father, norm_name, new_norm)
