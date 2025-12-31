import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, MinMaxObserver


class QRMSNorm(nn.Module):
    """
    RMSNorm with int16 per-tensor symmetric quantized weight.

    This implementation applies quantization to the weight tensor only,
    using per-tensor symmetric quantization with int16 range.
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        elementwise_affine=True,
        already_quantized_weight=False,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.already_quantized_w = already_quantized_weight

        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(normalized_shape, dtype=torch.bfloat16)
            )
        else:
            self.register_parameter("weight", None)

        # Weight quantization for int16 per-tensor symmetric
        self.weight_quant = FakeQuantize(
            observer=MinMaxObserver,
            quant_min=-32768,
            quant_max=32767,
            dtype=torch.qint32,
            qscheme=torch.per_tensor_symmetric,
        )

        self.w_q_cache = None
        self.use_weight_cache = already_quantized_weight

    def _clear_cache(self):
        self.w_q_cache = None

    def forward(self, x):
        # Compute RMS norm
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Apply quantized weight
        if self.weight is not None:
            if self.w_q_cache is not None:
                w_q = self.w_q_cache
            else:
                if self.already_quantized_w:
                    w_q = self.weight
                else:
                    w_q = self.weight_quant(self.weight)

                if self.use_weight_cache:
                    self.w_q_cache = w_q

            x = x * w_q

        return x
