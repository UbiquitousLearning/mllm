import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, PerChannelMinMaxObserver
from pymllm.backends.qualcomm.transformers.core.observer import (
    PerBlockParamFakeQuantize,
)


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
                isinstance(self.weight_quant, PerBlockParamFakeQuantize)
                and self.weight_quant is not None
            ):
                self.weight_quant.enable_observer()
                self.weight_quant.activation_post_process(self.weight)
                s, zp = self.weight_quant.activation_post_process.calculate_qparams()
                self.weight_quant.disable_observer()
                self.weight_quant.scale = s
                self.weight_quant.zero_point = zp
                print(
                    f"[{self.__class__.__name__}] Scale Shape: {list(s.shape)}, "
                    f"scale[:3]: {s.flatten()[:3].tolist()}, zp: {zp.flatten()[:3].tolist()}"
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

    @torch.no_grad()
    def convert_to_conv2d_deploy_hwio(self):
        """
        Convert to deploy format with HWIO layout [1, 1, In, Out].
        This format is commonly used by convolution-based inference engines.
        """
        if self.deploy_mode:
            return
        if self.weight_quant.scale is None:
            self.freeze_weight()

        scale = self.weight_quant.scale  # Shape: [Out]
        zero_point = self.weight_quant.zero_point  # Shape: [Out]

        # Step 1: Quantize in [Out, In] layout to ensure precision correctness
        w_q_obj = torch.quantize_per_channel(
            self.weight.float(), scale, zero_point, axis=0, dtype=torch.qint8
        )
        w_int = w_q_obj.int_repr()  # Shape: [Out, In]

        # Step 2: Critical step - Transpose and Reshape
        # [Out, In] -> Transpose -> [In, Out]
        w_transposed = w_int.t().contiguous()

        # [In, Out] -> [1, 1, In, Out] (HWIO)
        w_hwio = w_transposed.view(1, 1, self.in_features, self.out_features)

        # Step 3: Process Scale/ZP
        # Scale corresponds to Channel_Out, now at dimension 3 (index 3)
        # Reshape to [1, 1, 1, Out] for broadcasting
        scale_hwio = scale.view(1, 1, 1, self.out_features)
        zp_hwio = zero_point.view(1, 1, 1, self.out_features)

        # Step 4: Register buffers
        del self.weight
        self.register_buffer("weight", w_hwio)
        self.register_buffer("scale", scale_hwio)
        self.register_buffer("zero_point", zp_hwio)
        del self.weight_quant

        self.deploy_mode = True
        print(
            f"[{self.__class__.__name__}] Converted to HWIO. Weight: {self.weight.shape}"
        )


class QLinearLPBQ(QLinear):
    def __init__(self, in_features, out_features, bias=True, block_size=64):
        super().__init__(in_features, out_features, bias)
        self.block_size = [1, block_size]
        self.weight_quant = PerBlockParamFakeQuantize(
            dtype=torch.int8,
            quant_min=-7,
            quant_max=7,
            block_size=self.block_size,
            eps=0.0001 / 255,
            ch_axis=0,
        )

    def forward(self, x):
        # Must use quantized weights w_q for computation
        w_q = self.weight_quant(self.weight)
        return F.linear(x, w_q, self.bias)

    @torch.no_grad()
    def convert_to_conv2d_deploy_hwio(self):
        linear_scale = self.weight_quant.scale
        linear_zero_point = self.weight_quant.zero_point
        print(
            "Original Linear Scale[:3]: , zp[:3]: ",
            linear_scale.flatten()[:3].tolist(),
            linear_zero_point.flatten()[:3].tolist(),
        )

        # Convert weight to int4 (represent as int8)
        assert self.weight.shape[-1] % self.block_size[1] == 0
        weight_int4 = self.weight.reshape(self.out_features, -1, self.block_size[1])
        weight_int4 = weight_int4 / linear_scale.unsqueeze(-1)
        weight_int4 = weight_int4.reshape(self.out_features, -1)
        weight_int4 = weight_int4.round()
        assert weight_int4.min() >= -7 and weight_int4.max() <= 7
        weight_int4 = weight_int4.clamp(min=-7, max=7).to(torch.int8)

        # LPBQ Scale Quantization
        # Quantize fp32 scale to uint4 scale
        bitwidth_of_scale = 4
        num_channels = linear_scale.shape[0]  # [O, I / block_size[1]]
        num_steps = 2**bitwidth_of_scale
        quant_scales_dtype = torch.uint8
        quantized_scales = []
        level_2_scales = []
        for ch in range(num_channels):
            candidates = linear_scale[ch]
            max_scale = candidates.reshape(1, -1).amax(dim=-1) / num_steps
            q_scales = torch.clamp(
                input=torch.round(input=candidates / max_scale),
                min=1,
                max=2**bitwidth_of_scale,
            ).to(quant_scales_dtype)
            quantized_scales.append(q_scales)
            level_2_scales.append(max_scale)
        quantized_scales = torch.stack(
            quantized_scales, dim=0
        )  # [level 1, scale is uint4]
        level_2_scales = torch.stack(level_2_scales, dim=0)  # [level 2, scale is fp32]

        # Reformat Linear weight layout(OI) to Conv2d layout(HWIO,H=1,W=1)
        weight_int4 = (
            weight_int4.t()
            .contiguous()
            .view(1, 1, self.in_features, self.out_features)
            .contiguous()
        )

        del self.weight
        self.register_buffer("weight", weight_int4)
        self.register_buffer("scale1", quantized_scales.flatten())
        self.register_buffer("scale2", level_2_scales.flatten())
        del self.weight_quant
        self.deploy_mode = True
        print(
            f"[{self.__class__.__name__}] Converted to HWIO. Weight: {self.weight.shape}",
            f"Scale1(uint4): {self.scale1.shape}",
            f"Scale2(fp32): {self.scale2.shape}",
        )
