import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver


class QLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self.act_quant = None
        self.weight_quant = None
        self.w_q_cache = None

    def _setup_status(self, already_quantized_w, already_quantized_a):
        if self.act_quant:
            if already_quantized_a:
                self.act_quant.disable_observer()
            else:
                self.act_quant.enable_observer()
        if self.weight_quant:
            if already_quantized_w:
                self.weight_quant.disable_observer()
            else:
                self.weight_quant.enable_observer()

    def _clear_cache(self):
        self.w_q_cache = None


class QLinearW8A16_PerChannelSym_PerTensorSym(QLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        already_quantized_weight=False,
        already_quantized_activation=False,
    ):
        super().__init__(in_features, out_features, bias)

        self.weight_quant = FakeQuantize(
            observer=PerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint32,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        )
        self._setup_status(already_quantized_weight, already_quantized_activation)

    def forward(self, x):
        x_q = x
        if self.w_q_cache is not None:
            w_q = self.w_q_cache
        else:
            w_q = self.weight_quant(self.weight)
            self.w_q_cache = w_q
        return F.linear(x_q, w_q, self.bias)


class QLinearLPBQ(QLinear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        block_size=64,
        already_quantized_weight=False,
        already_quantized_activation=False,
    ):
        super().__init__(in_features, out_features, bias)

        self.block_size = block_size
        self.already_quantized_w = already_quantized_weight

        # Define buffers to store quantization parameters
        # Initially set to None, populated during first forward pass, or saved to state_dict
        self.register_buffer("scale_2_fp32", None)  # Level 2 Scale (FP32/BF16)
        self.register_buffer(
            "scale_1_uint4", None
        )  # Level 1 Scale Indices (Uint4 stored as Uint8)
        self.register_buffer("weight_q", None)  # Weight Indices (Int4 stored as Int8)

        self._setup_status(already_quantized_weight, already_quantized_activation)

    def _fake_quant_weight_double(self, w):
        """
        Double quantization calculation (no STE, forward-only simulation)
        And save quantization parameters to Buffer
        """
        out_channels, in_channels = w.shape

        # 1. Padding
        padding = 0
        if in_channels % self.block_size != 0:
            padding = self.block_size - (in_channels % self.block_size)
            w = F.pad(w, (0, padding), "constant", 0)

        # Reshape: [Out, Num_Blocks, Block_Size]
        w_reshaped = w.view(out_channels, -1, self.block_size)

        # =======================================================
        # Level 1 Scale Calculation (Ideal FP32)
        # =======================================================
        w_int4_max = 7.0
        # w_int4_min = -8.0

        # [Out, Num_Blocks, 1]
        w_abs_max = w_reshaped.abs().amax(dim=-1, keepdim=True)
        scale_1_fp32 = w_abs_max / w_int4_max
        scale_1_fp32 = torch.clamp(scale_1_fp32, min=1e-8)

        # =======================================================
        # Level 2 Scale Calculation & Level 1 Scale Quantization
        # =======================================================
        s_uint4_max = 15.0
        s_uint4_min = 0.0

        # Calculate Level 2 Scale (Per-Channel FP32) -> [Out, 1, 1]
        scale_2_fp32 = scale_1_fp32.amax(dim=1, keepdim=True) / s_uint4_max
        scale_2_fp32 = torch.clamp(scale_2_fp32, min=1e-8)

        # Quantize Level 1 Scale: FP32 -> Uint4 Indices
        scale_1_q = torch.round(scale_1_fp32 / scale_2_fp32)
        scale_1_q = torch.clamp(scale_1_q, s_uint4_min, s_uint4_max)

        # Dequantize Level 1 Scale
        scale_1_recon = scale_1_q * scale_2_fp32

        # =======================================================
        # Apply Level 1 Quantization (Quantize Weights)
        # =======================================================
        w_int4_min = -8.0

        # Quantize Weight: FP32 -> Int4 Indices
        w_q = torch.round(w_reshaped / scale_1_recon)
        w_q = torch.clamp(w_q, w_int4_min, w_int4_max)

        # Dequantize Weight
        w_recon = w_q * scale_1_recon

        # =======================================================
        # [NEW] Store Scales and Indices
        # =======================================================
        # Note: We store Indices here, typically converted to int8/uint8 to save space
        # scale_2 itself is a floating-point number, kept as is
        self.scale_2_fp32 = scale_2_fp32.detach()
        # scale_1_q is 0-15, stored as uint8
        self.scale_1_uint4 = scale_1_q.detach().to(torch.uint8)
        # w_q is -8 to 7, stored as int8
        self.weight_q = w_q.detach().to(torch.int8)

        # =======================================================
        # Restore Shape
        # =======================================================
        w_out = w_recon.view(out_channels, -1)
        if padding > 0:
            w_out = w_out[:, :-padding]

        return w_out.to(torch.bfloat16)

    def forward(self, x):
        x_q = x

        if self.w_q_cache is not None:
            w_q = self.w_q_cache
        else:
            if self.already_quantized_w:
                w_q = self.weight
            else:
                # Real-time calculation and update of self.scale_2, self.scale_1_idx, self.weight_idx
                w_q = self._fake_quant_weight_double(self.weight)

            if self.use_weight_cache:
                self.w_q_cache = w_q

        return F.linear(x_q, w_q, self.bias)
