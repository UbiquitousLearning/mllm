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
        self.quant_bits = quant_bits
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.weight = nn.Parameter(torch.ones(normalized_shape))

        # Quantization configuration for Weight
        self.weight_fake_quant = FakeQuantize(
            observer=MinMaxObserver.with_args(
                qscheme=torch.per_tensor_affine, dtype=torch.qint32
            ),
            quant_min=0,
            quant_max=2 ** (quant_bits) - 1,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_affine,
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
    def convert_to_deploy(self):
        """
        In-place replacement of self.weight:
        Float Parameter -> Int Buffer
        """
        # 1. Ensure quantization parameters are ready
        if self.weight_fake_quant.scale is None:
            self.freeze_weight()

        scale = self.weight_fake_quant.scale
        zero_point = self.weight_fake_quant.zero_point
        quant_min = self.weight_fake_quant.quant_min
        quant_max = self.weight_fake_quant.quant_max

        # 2. Calculate integer values
        # w_int = round(w / s + zp)
        w_int = torch.round(self.weight / scale + zero_point).clamp(
            quant_min, quant_max
        )

        # 3. Set target integer type
        if self.quant_bits <= 8:
            target_dtype = torch.int8
        elif self.quant_bits <= 16:
            target_dtype = torch.int16
        else:
            target_dtype = torch.int32

        w_int = w_int.to(target_dtype)

        # === Key steps: Replacement operations ===

        # A. Delete original Parameter 'weight'
        # Must delete first, otherwise cannot register buffer with same name
        del self.weight

        # B. Register Buffer with same name 'weight'
        # This makes state_dict['weight'] become Int Tensor
        self.register_buffer("weight", w_int)

        # C. Register Scale (usually needed by engine)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

        # D. Clean up unnecessary modules
        if hasattr(self, "weight_fake_quant"):
            del self.weight_fake_quant

        class_name = self.__class__.__name__
        instance_class_name = type(self).__name__
        print(
            f"Class: {class_name}, Instance: {instance_class_name}, Deploy Mode Activated. 'weight' is now {self.weight.dtype} buffer. zp is {zero_point}"
        )

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
