import torch
import torch.nn as nn
from torch.ao.quantization import (
    FakeQuantize,
    MinMaxObserver,
)
from torch.ao.quantization.observer import FixedQParamsObserver

DEFAULT_EPS_8BIT = 0.0001 / 255
DEFAULT_EPS_16BIT = 0.0001 / 65535


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
            # NOTE: If left empty: with uint8 and symmetric quantization, the observer will use [0, 255] as the range. And 128 as the zero_point.
            self.quant_min = None
            self.quant_max = None
            assert bits == 8, "Symmetric quantization is only supported for 8-bit"
            self.dtype = torch.uint8
        else:
            # Asymmetric (Affine): range is [0, 2^bits - 1]
            # e.g., 8-bit: 0 to 255
            self.quant_min = 0
            self.quant_max = (2**bits) - 1

        if bits == 8:
            eps = DEFAULT_EPS_8BIT
        elif bits == 16:
            eps = DEFAULT_EPS_16BIT
        else:
            raise ValueError(f"Unsupported bit width: {bits}")

        # 2. Initialize FakeQuantize
        # MovingAverageMinMaxObserver calculates scale and zero_point based on observed tensors.
        # Passing quant_min/max to the observer ensures consistency.
        self.fake_quant = FakeQuantize(
            observer=MinMaxObserver.with_args(
                dtype=self.dtype,
                qscheme=self.qscheme,
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                reduce_range=False,
                eps=eps,
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
        self.fake_quant.enable_fake_quant()

    def disable_fakequant(self):
        """Disable quantization simulation (act as identity)."""
        self.fake_quant.disable_fake_quant()

    def extra_repr(self):
        mode = "Symmetric" if "symmetric" in str(self.qscheme) else "Asymmetric"
        return f"bits={self.bits}, mode={mode}, q_range=({self.quant_min}, {self.quant_max}), dtype={self.dtype}"


class FixedActivationQDQ(nn.Module):
    """
    Fixed activation Quantization-DeQuantization (QDQ) module.
    Uses pre-determined scale and zero_point instead of dynamic observation.
    Supports both Symmetric and Asymmetric (Affine) quantization.
    Uses torch.qint32 as a unified type to support various bit-widths.
    """

    def __init__(self, scale, zero_point, bits=8, qscheme=torch.per_tensor_affine):
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

        if bits not in [8, 16]:
            raise ValueError(f"Unsupported bit width: {bits}")

        # 2. Convert scale and zero_point to tensors if needed
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)
        if not isinstance(zero_point, torch.Tensor):
            zero_point = torch.tensor(zero_point, dtype=torch.int32)

        # 3. Initialize FakeQuantize with fixed parameters
        # Use FakeQuantize with FixedQParamsObserver for fixed scale and zero_point
        self.fake_quant = FakeQuantize.with_args(
            observer=FixedQParamsObserver.with_args(
                scale=scale,
                zero_point=zero_point,
            ),
            dtype=self.dtype,
            qscheme=self.qscheme,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
        )()

    def forward(self, x):
        # Applies fake quantization with fixed scale and zero_point:
        # rounds to nearest integer and clamps to [min, max],
        # then dequantizes back to float to simulate quantization noise.
        return self.fake_quant(x)

    # Control methods for quantization-aware training (QAT)
    # Note: FixedActivationQDQ doesn't have observer, so these methods
    # only control fake quantization behavior
    def enable_observer(self):
        """No-op: FixedActivationQDQ doesn't use observer."""
        pass

    def disable_observer(self):
        """No-op: FixedActivationQDQ doesn't use observer."""
        pass

    def enable_fakequant(self):
        """Enable simulation of quantization error."""
        self.fake_quant.enable_fake_quant()

    def disable_fakequant(self):
        """Disable quantization simulation (act as identity)."""
        self.fake_quant.disable_fake_quant()

    @property
    def scale(self):
        """Get the fixed scale value."""
        return self.fake_quant.scale

    @property
    def zero_point(self):
        """Get the fixed zero_point value."""
        return self.fake_quant.zero_point

    def extra_repr(self):
        mode = "Symmetric" if "symmetric" in str(self.qscheme) else "Asymmetric"
        scale_val = self.scale.item() if self.scale.numel() == 1 else self.scale
        zp_val = (
            self.zero_point.item() if self.zero_point.numel() == 1 else self.zero_point
        )
        return f"bits={self.bits}, mode={mode}, scale={scale_val}, zero_point={zp_val}, q_range=({self.quant_min}, {self.quant_max}), dtype={self.dtype}"
