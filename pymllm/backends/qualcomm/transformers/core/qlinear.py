import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, PerChannelMinMaxObserver


class QLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.act_quant = None
        self.weight_quant = None
        self.deploy_mode = False

    @torch.no_grad()
    def freeze_weight(self):
        """PTQ Core: Observe current weights, calculate and fix Scale/ZP"""
        if self.weight_quant is not None:
            # Compatible with official FakeQuantize module
            if (
                isinstance(self.weight_quant, FakeQuantize)
                and self.weight_quant is not None
            ):
                _ = self.weight_quant(self.weight)
                self.weight_quant.disable_observer()
                s = self.weight_quant.scale
                print(
                    f"[{self.__class__.__name__}] Scale Shape: {list(s.shape)}, "
                    f"scale[:3]: {s.flatten()[:3].tolist()}"
                )
            # Compatible with custom LPBQ logic
            elif hasattr(self.weight_quant, "freeze"):
                self.weight_quant.freeze(self.weight.detach())
                s = self.weight_quant.scale_2_fp32
                if s is not None:
                    print(
                        f"[{self.__class__.__name__}] LPBQ L2 Scale Shape: {list(s.shape)}, "
                        f"scale[:3]: {s.flatten()[:3].tolist()}"
                    )

    def forward(self, x):
        raise NotImplementedError


# --- 1. W8A16 Per-Channel Scheme ---
class QLinearW8A16_PerChannelSym(QLinear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)

        # Weight: Int8 Per-Channel symmetric
        self.weight_quant = FakeQuantize(
            observer=PerChannelMinMaxObserver.with_args(
                qscheme=torch.per_channel_symmetric,
                dtype=torch.qint8,
                ch_axis=0,  # Quantize output channels
            ),
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
        )

    def forward(self, x):
        assert self.deploy_mode is False
        # Activation quantization logic (add act_quant here if needed)
        x_q = x
        # Apply fake quantization: use fixed scale if frozen, otherwise update in real-time
        w_q = self.weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)

    @torch.no_grad()
    def convert_to_deploy(self):
        if self.deploy_mode:
            return

        # 1. Ensure Observer is frozen
        if self.weight_quant.scale is None:
            self.freeze_weight()

        scale = self.weight_quant.scale
        zero_point = self.weight_quant.zero_point

        # 2. Use PyTorch native API for Per-Channel quantization
        # This handles per-channel complexity and returns quantized tensor
        w_q_obj = torch.quantize_per_channel(
            self.weight.float(), scale, zero_point, axis=0, dtype=torch.qint8
        )

        # 3. Extract pure integer data
        w_int = w_q_obj.int_repr()

        # 4. Replace Parameter with Buffer
        del self.weight
        # Register buffer named 'weight' to maintain name consistency
        self.register_buffer("weight", w_int)
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)

        # Remove fake quant module to reduce model size
        del self.weight_quant

        self.deploy_mode = True
        print(
            f"[{self.__class__.__name__}] Converted to deploy. Weight shape: {self.weight.shape}, dtype: {self.weight.dtype}"
        )


# --- 2. LPBQ (Double Quantization) Scheme ---
class DoubleQuantizer(nn.Module):
    """
    Handles LPBQ double normalization logic to work like FakeQuantize
    """

    def __init__(self, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.register_buffer("is_frozen", torch.tensor(False))
        self.register_buffer("scale_2_fp32", None)
        self.register_buffer("scale_1_uint4", None)
        self.register_buffer("weight_q", None)
        self.w_recon_cached = None  # Cache dequantized weights for acceleration

    def freeze(self, w):
        # Run complete double quantization and store in buffer
        self.w_recon_cached = self.quantize_dequantize(w, save_buffers=True)
        self.is_frozen = torch.tensor(True)

    def quantize_dequantize(self, w, save_buffers=False):
        out_channels, in_channels = w.shape
        # 1. Padding handling
        pad_len = (self.block_size - in_channels % self.block_size) % self.block_size
        if pad_len > 0:
            w = F.pad(w, (0, pad_len), "constant", 0)

        w_reshaped = w.view(out_channels, -1, self.block_size)

        # Level 1: FP32 Scale
        s1 = w_reshaped.abs().amax(dim=-1, keepdim=True) / 7.0
        s1 = s1.clamp(min=1e-8)

        # Level 2: Quantize S1 to Uint4
        s2 = s1.amax(dim=1, keepdim=True) / 15.0
        s2 = s2.clamp(min=1e-8)
        s1_q = (s1 / s2).round().clamp(0, 15)
        s1_recon = s1_q * s2

        # Level 3: Quantize Weight to Int4
        w_q = (w_reshaped / s1_recon).round().clamp(-8, 7)
        w_recon = w_q * s1_recon

        if save_buffers:
            self.scale_2_fp32 = s2.detach()
            self.scale_1_uint4 = s1_q.detach().to(torch.uint8)
            self.weight_q = w_q.detach().to(torch.int8)

        # Restore shape
        w_out = w_recon.view(out_channels, -1)
        if pad_len > 0:
            w_out = w_out[:, :-pad_len]
        return w_out

    def forward(self, w):
        if self.is_frozen:
            # If frozen, directly return cached reconstructed weights (or real-time dequantization from Buffer)
            if self.w_recon_cached is None:
                # Logic to reconstruct from weight_q + scale_1 + scale_2 can be written here
                pass
            return (
                self.w_recon_cached
                if self.w_recon_cached is not None
                else self.quantize_dequantize(w)
            )
        return self.quantize_dequantize(w)


class QLinearLPBQ(QLinear):
    def __init__(self, in_features, out_features, bias=True, block_size=64):
        super().__init__(in_features, out_features, bias)
        self.weight_quant = DoubleQuantizer(block_size)

    def forward(self, x):
        # Must use quantized weights w_q for computation
        w_q = self.weight_quant(self.weight)
        return F.linear(x, w_q, self.bias)

    @torch.no_grad()
    def convert_to_deploy(self):
        if self.deploy_mode:
            return

        del self.weight
        self.register_buffer(
            "weight",
            self.weight_quant.weight_q.reshape(self.weight_quant.weight_q.shape[0], -1),
        )
        self.register_buffer("scale1", self.weight_quant.scale_1_uint4)
        self.register_buffer("scale2", self.weight_quant.scale_2_fp32)
        del self.weight_quant

        self.deploy_mode = True
        print(
            f"[{self.__class__.__name__}] Converted to deploy. Original float weight removed."
        )
