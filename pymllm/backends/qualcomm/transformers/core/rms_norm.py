import torch
import torch.nn as nn
from torch.ao.quantization import FakeQuantize, MinMaxObserver


class QRMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        quant_bits=16,
    ):
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.weight = nn.Parameter(torch.ones(normalized_shape))

        # Quantization configuration for Weight
        self.weight_fake_quant = FakeQuantize(
            observer=MinMaxObserver.with_args(
                qscheme=torch.per_tensor_symmetric, dtype=torch.qint32
            ),
            quant_min=-(2 ** (quant_bits - 1)),
            quant_max=2 ** (quant_bits - 1) - 1,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_symmetric,
        )

    def forward(self, x):
        # 1. RMSNorm basic logic (using float32 to ensure stability)
        input_dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)

        # 2. Weight fake quantization
        # If observer is not closed, this step will continuously update scale/zp
        # If freeze_weight() is called, this will just use fixed scale/zp for quantization
        w_q = self.weight_fake_quant(self.weight)

        return (x_normed * w_q).to(input_dtype)

    @torch.no_grad()
    def freeze_weight(self):
        """
        Manually trigger Observer to observe and calculate scale, then lock it.
        Solve the problem of output being 0 on first run.
        """
        self.weight_fake_quant.activation_post_process(self.weight)
        s, zp = self.weight_fake_quant.activation_post_process.calculate_qparams()
        self.weight_fake_quant.scale.copy_(s)
        self.weight_fake_quant.zero_point.copy_(zp)
        self.weight_fake_quant.disable_observer()
        class_name = self.__class__.__name__
        instance_class_name = type(self).__name__
        print(
            f"Class: {class_name}, Instance: {instance_class_name}, Weight Quantized: scale={self.weight_fake_quant.scale}, zp={self.weight_fake_quant.zero_point}"
        )

    def disable_quant(self):
        """Completely turn off quantization noise and return to floating point mode"""
        self.weight_fake_quant.disable_fakequant()

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
