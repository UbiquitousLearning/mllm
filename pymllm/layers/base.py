import torch
from torch import nn
from torch.nn import Parameter
from pymllm.layers.utils import set_weight_attrs
from pymllm.quantization.quant_recipe import QuantRecipe


class MllmBaseLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_recipe: QuantRecipe = None

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        """Load weights into a parameter.

        This is the default implementation that directly copies the loaded weight
        into the parameter. Subclasses should override this method to implement
        custom loading logic (e.g., tensor parallelism sharding).

        Args:
            param: The parameter to load weights into.
            loaded_weight: The weight tensor loaded from checkpoint.
        """
        param.data.copy_(loaded_weight)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")
