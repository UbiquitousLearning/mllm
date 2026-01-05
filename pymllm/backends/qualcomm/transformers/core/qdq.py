import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver


class ActivationQDQ(nn.Module):
    """
    General activation Quantization-DeQuantization (QDQ) module.
    Supports both Symmetric and Asymmetric (Affine) quantization.
    Uses torch.qint32 as a unified type to support various bit-widths.
    """

    def __init__(self, bits=8, qscheme=torch.per_tensor_affine):
        super().__init__()
        self.bits = bits
        self.qscheme = qscheme

        # Define the simulation dtype as qint32 to avoid overflow across different bit-widths
        self.dtype = torch.qint32

        # 1. Calculate quantization range based on bits and scheme
        if qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]:
            # Symmetric: range is [-(2^(bits-1)), 2^(bits-1) - 1]
            # e.g., 8-bit: -128 to 127
            self.quant_min = -(2 ** (bits - 1))
            self.quant_max = 2 ** (bits - 1) - 1
        else:
            # Asymmetric (Affine): range is [0, 2^bits - 1]
            # e.g., 8-bit: 0 to 255
            self.quant_min = 0
            self.quant_max = (2**bits) - 1

        # 2. Initialize FakeQuantize
        # MinMaxObserver calculates scale and zero_point based on observed tensors.
        # Passing quant_min/max to the observer ensures consistency.
        self.fake_quant = FakeQuantize(
            observer=MinMaxObserver.with_args(
                qscheme=self.qscheme,
                dtype=self.dtype,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                reduce_range=False,
            ),
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            dtype=self.dtype,
            qscheme=self.qscheme,
        )

    def forward(self, x):
        # Applies fake quantization: rounds to nearest integer and clamps to [min, max],
        # then dequantizes back to float to simulate quantization noise.
        return self.fake_quant(x)

    # Control methods for quantization-aware training (QAT)
    def enable_observer(self):
        """Enable tracking of min/max values to update scale and zero_point."""
        self.fake_quant.enable_observer()

    def disable_observer(self):
        """Freeze scale and zero_point calculation."""
        self.fake_quant.disable_observer()

    def enable_fakequant(self):
        """Enable simulation of quantization error."""
        self.fake_quant.enable_fakequant()

    def disable_fakequant(self):
        """Disable quantization simulation (act as identity)."""
        self.fake_quant.disable_fakequant()

    def extra_repr(self):
        mode = "Symmetric" if "symmetric" in str(self.qscheme) else "Asymmetric"
        return f"bits={self.bits}, mode={mode}, q_range=({self.quant_min}, {self.quant_max}), dtype={self.dtype}"
