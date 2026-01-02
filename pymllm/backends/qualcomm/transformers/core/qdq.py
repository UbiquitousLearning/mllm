import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver


class ActivationQDQ(nn.Module):
    """
    General activation value pseudo-quantization module (QDQ).
    Supports symmetric Per-Tensor quantization, configurable bit numbers (e.g., 8-bit or 16-bit).
    """

    def __init__(self, bits=8, qscheme=torch.per_tensor_symmetric):
        super().__init__()

        # 1. Calculate quantization range based on bits
        # int8: -128 to 127
        # int16: -32768 to 32767
        self.quant_min = -(2 ** (bits - 1))
        self.quant_max = 2 ** (bits - 1) - 1

        # 2. Initialize FakeQuantize
        # For activations, typically use MinMaxObserver or MovingAverageMinMaxObserver
        self.fake_quant = FakeQuantize(
            observer=MinMaxObserver.with_args(qscheme=qscheme, dtype=torch.qint32),
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=torch.qint32,
            qscheme=qscheme,
        )

    def forward(self, x):
        # Directly apply pseudo-quantization.
        # When observer is enabled, it continuously updates scale/zp;
        # When fakequant is enabled, it simulates quantization errors.
        return self.fake_quant(x)

    def enable_observer(self):
        self.fake_quant.enable_observer()

    def disable_observer(self):
        self.fake_quant.disable_observer()

    def enable_fakequant(self):
        self.fake_quant.enable_fakequant()

    def disable_fakequant(self):
        self.fake_quant.disable_fakequant()

    def extra_repr(self):
        return f"bits={self.quant_max.bit_length() + 1}, q_range=({self.quant_min}, {self.quant_max})"
