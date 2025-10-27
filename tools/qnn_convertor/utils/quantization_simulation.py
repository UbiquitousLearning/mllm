"""
This file performs a fake quantization of the model weights and activations.
It will be used to simulate the quantization of the model weights and activations.
"""

import torch
from torch import nn
from functools import partial
import numpy as np


@torch.no_grad()
def simulate_quantize_weight_per_channel_absmax(w, n_bits=8):
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w


@torch.no_grad()
def simulate_quantize_weight_per_tensor_absmax(w, n_bits=8):
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().clamp_((-2**(n_bits-1)), (2**(n_bits-1)-1)).mul_(scales)
    return w


# @torch.no_grad()
# def simulate_quantize_weight_scale_per_tensor_absmax(w, n_bits=8):
#     scales = w.abs().max()
#     q_max = 2 ** (n_bits - 1) - 1
#     scales.clamp_(min=1e-5).div_(q_max)
#     return scales


@torch.no_grad()
def simulate_quantize_activation_per_tensor_static_input(t, scale=1, n_scale_bits = 8, n_bits=8, clip_top=False):
    assert n_scale_bits <= n_bits

    scale = scale.clone().to(t.device)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_scale_bits - 1) - 1
    scale.clamp_(min=1e-5).div_(q_max)
    t = t.div(scale).round()
    if clip_top:
        t = t.clamp(-2**(n_bits-1), 2**(n_bits-1) - 1)
    t = t.mul(scale)
    return t


@torch.no_grad()
def simulate_quantize_activation_per_tensor_static_output(t, scale=1, n_scale_bits = 8, n_bits=8, clip_top=False):
    assert n_scale_bits <= n_bits

    scale = scale.clone().to(t.device)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_scale_bits - 1) - 1
    scale.clamp_(min=1e-5).div_(q_max)

    t = t.div(scale)
    t = t.round()
    if clip_top:
        t = t.clamp(-2**(n_bits-1), 2**(n_bits-1) - 1)
    t = t.mul(scale)
    return t


@torch.no_grad()
def simulate_quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


@torch.no_grad()
def simulate_quantize_activation_per_tensor_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    return t


class W8A8LinearStatic(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        input_scale,
        output_scale,
        bias=True,
        clip_top=False,
        name="default"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_scale = torch.tensor(input_scale)
        self.output_scale = torch.tensor(output_scale)

        self.weight_scale = None
        self.weight_quant_type = None

        self.printAct = False

        self.name= name

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        self.act_quant_input = partial(
            simulate_quantize_activation_per_tensor_static_input,
            n_bits=8,
            clip_top=clip_top["input"],
        )
        self.act_quant_output = partial(
            simulate_quantize_activation_per_tensor_static_output,
            n_bits=8,
            clip_top=clip_top["output"],
        )

    def to(self, *args, **kwargs):
        super(W8A8LinearStatic, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        # perform online quantize-dequantize matmul to simulate W8A8 inference
        # if self.name != "default":
        #     print(self.name)
        #     # dump output q_y to file
        #     cpu_q_y = x.to('cpu').float().squeeze(0)
        #     # print(self.name, cpu_q_y.shape)
        #     np.savetxt(self.name + "—input.csv", cpu_q_y.numpy(), delimiter=',', fmt='%.6f')

        q_x = self.act_quant_input(x, scale=self.input_scale)

        # if self.name != "default":
        #     print(self.name)
        #     print(q_x.shape)
        #     # dump output q_y to file

        #     cpu_q_y = (q_x / (self.input_scale.clamp(min=1e-5).div(127))).to('cpu').int().squeeze(0)
        #     # print(self.name, cpu_q_y.shape)
        #     np.savetxt(self.name + "—input-int8.csv", cpu_q_y.numpy(), delimiter=',', fmt='%.6f')
        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.act_quant_output(y, scale=self.output_scale)

        # if self.printAct:
        #     print(q_y)
        
        # if self.name != "default":
        #     print(self.name)
        #     # dump output q_y to file
        #     cpu_q_y = q_y.to('cpu').float().squeeze(0)
        #     print(self.name, cpu_q_y.shape)
        #     np.savetxt(self.name + ".csv", cpu_q_y.numpy(), delimiter=',', fmt='%.6f')

        return q_y

    @staticmethod
    def from_float(module, scales, weight_quant_type="per_tensor", clip_top=False, name="default"):
        assert isinstance(module, torch.nn.Linear)

        new_module = W8A8LinearStatic(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            input_scale=scales["input"],
            output_scale=scales["output"],
            clip_top=clip_top,
            name=name
        )

        if weight_quant_type == "per_channel":
            new_module.weight = simulate_quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant_type == "per_tensor":
            new_module.weight = simulate_quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant_type}")

        new_module.weight_quant_name = weight_quant_type

        if module.bias is not None:
            if name != "default":
                print(name)
                # dump output q_y to file
                cpu_bias = module.bias.to('cpu').float()
                print(name, cpu_bias.shape)
                np.savetxt(name + "-bias.csv", cpu_bias.numpy(), delimiter=',', fmt='%.6f')
            new_module.bias = simulate_quantize_weight_per_tensor_absmax(module.bias, n_bits=8)

        return new_module
    

    @staticmethod
    def from_float_and_print(module, scales, weight_quant_type="per_tensor", clip_top=False):
        assert isinstance(module, torch.nn.Linear)

        new_module = W8A8LinearStatic(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            input_scale=scales["input"],
            output_scale=scales["output"],
            clip_top=clip_top,
        )

        if weight_quant_type == "per_channel":
            new_module.weight = simulate_quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant_type == "per_tensor":
            new_module.weight = simulate_quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant_type}")

        new_module.weight_quant_name = weight_quant_type
        new_module.printAct = True

        if module.bias is not None:
            new_module.bias = simulate_quantize_weight_per_tensor_absmax(module.bias, n_bits=8)

        return new_module

    def __repr__(self):
        return f"W8A8LinearStatic({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, input_scale={self.input_scale.item()}, output_scale={self.output_scale.item()}, clip_top={self.act_quant_input.keywords['clip_top']})"
  


class W8AX_LinearStatic(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        input_scale,
        output_scale,
        n_scale_bits,
        n_bits,
        bias=True,
        clip_top=False,
        name="default"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_scale = torch.tensor(input_scale)
        self.output_scale = torch.tensor(output_scale)

        self.weight_scale = None
        self.weight_quant_type = None

        self.printAct = False

        self.name = name

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        self.act_quant_input = partial(
            simulate_quantize_activation_per_tensor_static_input,
            n_scale_bits=n_scale_bits,
            n_bits=n_bits,
            clip_top=clip_top["input"],
        )
        self.act_quant_output = partial(
            simulate_quantize_activation_per_tensor_static_output,
            n_scale_bits=n_scale_bits,
            n_bits=n_bits,
            clip_top=clip_top["output"],
        )

    def to(self, *args, **kwargs):
        super(W8AX_LinearStatic, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        # perform online quantize-dequantize matmul to simulate W8A8 inference

        # if self.name != "default":
        #     print(self.name)
        #     print(x.shape)
        #     # dump output q_y to file

        #     print(self.input_scale)

        #     cpu_q_y = x.to('cpu').float().squeeze(0)
        #     # print(self.name, cpu_q_y.shape)
        #     np.savetxt(self.name + "—input-beforequant.csv", cpu_q_y.numpy(), delimiter=',', fmt='%.6f')

        q_x = self.act_quant_input(x, scale=self.input_scale)


        # if self.name != "default":
        #     print(self.name)
        #     print(q_x.shape)
        #     # dump output q_y to file

        #     cpu_q_y = (q_x / (self.input_scale.clamp(min=1e-5).div(127))).to('cpu').int().squeeze(0)
        #     # print(self.name, cpu_q_y.shape)
        #     np.savetxt(self.name + "—input-int8.csv", cpu_q_y.numpy(), delimiter=',', fmt='%.6f')


        y = torch.functional.F.linear(q_x, self.weight, self.bias)
        q_y = self.act_quant_output(y, scale=self.output_scale)

        # if self.printAct:
        #     print(q_y)

        # if self.name != "default":
        #     print(self.name)
        #     # dump output q_y to file
        #     cpu_q_y = q_y.to('cpu').float().squeeze(0)
        #     np.savetxt(self.name + ".csv", cpu_q_y.numpy(), delimiter=',', fmt='%.6f')

        # if self.name != "default":
        #     print(self.name)
        #     # dump output q_y to file
        #     cpu_q_y = (q_y/(self.output_scale/127.0)).to('cpu').float().squeeze(0)
        #     np.savetxt(self.name + "-output-int8.csv", cpu_q_y.numpy(), delimiter=',', fmt='%.6f')
        return q_y

    @staticmethod
    def from_float(module, scales, n_scale_bits, n_bits, weight_quant_type="per_tensor", clip_top=False, name="default"):
        assert isinstance(module, torch.nn.Linear)

        new_module = W8AX_LinearStatic(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            input_scale=scales["input"],
            output_scale=scales["output"],
            clip_top=clip_top,
            n_scale_bits=n_scale_bits,
            n_bits=n_bits,
            name=name
        )

        if weight_quant_type == "per_channel":
            new_module.weight = simulate_quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant_type == "per_tensor":
            new_module.weight = simulate_quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant_type}")

        if module.bias is not None:
            # if name != "default":
            #     print(name)
            #     # dump output q_y to file
            #     cpu_bias = module.bias.clone().to('cpu').float()
            #     print(name, cpu_bias.shape)
            #     np.savetxt(name + "-bias.csv", cpu_bias.detach().numpy(), delimiter=',', fmt='%.6f')
            new_module.bias = simulate_quantize_weight_per_tensor_absmax(module.bias, n_bits=8)

        return new_module

    def __repr__(self):
        return f"W8AXLinearStatic({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, input_scale={self.input_scale.item()}, output_scale={self.output_scale.item()}, clip_top={self.act_quant_input.keywords['clip_top']})"

class W8AX_Nobias_LinearStatic(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        input_scale,
        output_scale,
        n_scale_bits,
        n_bits,
        bias=True,
        clip_top=False,
        name="default"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_scale = torch.tensor(input_scale)
        self.output_scale = torch.tensor(output_scale)

        self.weight_scale = None
        self.weight_quant_type = None

        self.printAct = False

        self.name = name

        self.register_buffer(
            "weight",
            torch.randn(
                self.out_features,
                self.in_features,
                dtype=torch.float16,
                requires_grad=False,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (1, self.out_features), dtype=torch.float16, requires_grad=False
                ),
            )
        else:
            self.register_buffer("bias", None)

        self.act_quant_input = partial(
            simulate_quantize_activation_per_tensor_static_input,
            n_scale_bits=n_scale_bits,
            n_bits=n_bits,
            clip_top=clip_top["input"],
        )
        self.act_quant_output = partial(
            simulate_quantize_activation_per_tensor_static_output,
            n_scale_bits=n_scale_bits,
            n_bits=n_bits,
            clip_top=clip_top["output"],
        )

    def to(self, *args, **kwargs):
        super(W8AX_LinearStatic, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        

        q_x = self.act_quant_input(x, scale=self.input_scale)

        y = torch.functional.F.linear(q_x, self.weight, None)
        q_y = self.act_quant_output(y, scale=self.output_scale)

        if  self.bias is not None:
            q_y = q_y + self.bias

        return q_y

    @staticmethod
    def from_float(module, scales, n_scale_bits, n_bits, weight_quant_type="per_tensor", clip_top=False, name="default"):
        assert isinstance(module, torch.nn.Linear)

        new_module = W8AX_Nobias_LinearStatic(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            input_scale=scales["input"],
            output_scale=scales["output"],
            clip_top=clip_top,
            n_scale_bits=n_scale_bits,
            n_bits=n_bits,
            name=name
        )

        if weight_quant_type == "per_channel":
            new_module.weight = simulate_quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant_type == "per_tensor":
            new_module.weight = simulate_quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant_type}")

        if module.bias is not None:
            
            new_module.bias = module.bias

        return new_module

    def __repr__(self):
        return f"W8AXLinearStatic({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, input_scale={self.input_scale.item()}, output_scale={self.output_scale.item()}, clip_top={self.act_quant_input.keywords['clip_top']})"


def quantize_opt(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=True,
    layer_clip={},
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():

        if isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8LinearStatic.from_float(
                m.fc1,
                decoder_scales["model." + name + ".fc1"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".fc1"],
            )
            m.fc2 = W8A8LinearStatic.from_float(
                m.fc2,
                decoder_scales["model." + name + ".fc2"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".fc2"],
            )
        elif isinstance(m, OPTAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales["model." + name + ".q_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".q_proj"],
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales["model." + name + ".k_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".k_proj"],
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales["model." + name + ".v_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".v_proj"],
            )
            m.out_proj = W8A8LinearStatic.from_float(
                m.out_proj,
                decoder_scales["model." + name + ".out_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".out_proj"],
            )
    return model


def quantize_llama_like(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=False,
    layer_clip={},
):
    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaMLP,
    )

    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            m.gate_proj = W8A8LinearStatic.from_float(
                m.gate_proj,
                decoder_scales["model." + name + ".gate_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".gate_proj"],
            )
            m.up_proj = W8A8LinearStatic.from_float(
                m.up_proj,
                decoder_scales["model." + name + ".up_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".up_proj"],
            )
            m.down_proj = W8A8LinearStatic.from_float(
                m.down_proj,
                decoder_scales["model." + name + ".down_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".down_proj"],
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales["model." + name + ".q_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".q_proj"],
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales["model." + name + ".k_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".k_proj"],
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales["model." + name + ".v_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".v_proj"],
            )
            m.o_proj = W8A8LinearStatic.from_float(
                m.o_proj,
                decoder_scales["model." + name + ".o_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".o_proj"],
            )
    return model


def quantize_qwen2_like(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=False,
    layer_clip={},
):
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, Qwen2MLP):
            m.gate_proj = W8A8LinearStatic.from_float(
                m.gate_proj,
                decoder_scales["model." + name + ".gate_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".gate_proj"],
                name = "model." + name + ".gate_proj",
            )
            m.up_proj = W8A8LinearStatic.from_float(
                m.up_proj,
                decoder_scales["model." + name + ".up_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".up_proj"],
                name = "model." + name + ".up_proj",
            )
            m.down_proj = W8A8LinearStatic.from_float(
                m.down_proj,
                decoder_scales["model." + name + ".down_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".down_proj"],
                name = "model." + name + ".down_proj",
            )
        elif isinstance(m, Qwen2Attention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            # print name of module m
            print(name)
            m.q_proj = W8AX_LinearStatic.from_float(
                m.q_proj,
                decoder_scales["model." + name + ".q_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".q_proj"],
                name = "model." + name + ".q_proj",
                n_scale_bits = 16,
                n_bits = 16,
            )
            m.k_proj = W8AX_LinearStatic.from_float(
                m.k_proj,
                decoder_scales["model." + name + ".k_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".k_proj"],
                name = "model." + name + ".k_proj",
                n_scale_bits = 16,
                n_bits = 16,
            )
            m.v_proj = W8AX_LinearStatic.from_float(
                m.v_proj,
                decoder_scales["model." + name + ".v_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".v_proj"],
                name = "model." + name + ".v_proj",
                n_scale_bits = 16,
                n_bits = 16,
            )
            m.o_proj = W8A8LinearStatic.from_float(
                m.o_proj,
                decoder_scales["model." + name + ".o_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".o_proj"],
                name = "model." + name + ".o_proj",
            )
    return model

def quantize_qwen2vl_like(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=False,
    layer_clip={},
):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLSdpaAttention,
        Qwen2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, Qwen2MLP):
            m.gate_proj = W8AX_LinearStatic.from_float(
                m.gate_proj,
                decoder_scales["model." + name + ".gate_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".gate_proj"],
                n_scale_bits = 8,
                n_bits = 8,
            )
            m.up_proj = W8AX_LinearStatic.from_float(
                m.up_proj,
                decoder_scales["model." + name + ".up_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".up_proj"],
                n_scale_bits = 8,
                n_bits = 8,
            )
            m.down_proj = W8AX_LinearStatic.from_float(
                m.down_proj,
                decoder_scales["model." + name + ".down_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".down_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name = "model." + name + ".down_proj",
            )
        elif isinstance(m, Qwen2VLSdpaAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8AX_LinearStatic.from_float(
                m.q_proj,
                decoder_scales["model." + name + ".q_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".q_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name= "model." + name + ".q_proj",
            )
            m.k_proj = W8AX_LinearStatic.from_float(
                m.k_proj,
                decoder_scales["model." + name + ".k_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".k_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name= "model." + name + ".k_proj",
            )
            m.v_proj = W8AX_LinearStatic.from_float(
                m.v_proj,
                decoder_scales["model." + name + ".v_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".v_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name= "model." + name + ".v_proj",
            )
            
            m.o_proj = W8AX_LinearStatic.from_float(
                m.o_proj,
                decoder_scales["model." + name + ".o_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".o_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name= "model." + name + ".o_proj",
            )
    return model


def quantize_qwen2vl_woqkv_like(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=False,
    layer_clip={},
):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLSdpaAttention,
        Qwen2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, Qwen2MLP):
            m.gate_proj = W8AX_LinearStatic.from_float(
                m.gate_proj,
                decoder_scales["model." + name + ".gate_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".gate_proj"],
                n_scale_bits = 8,
                n_bits = 8,
            )
            m.up_proj = W8AX_LinearStatic.from_float(
                m.up_proj,
                decoder_scales["model." + name + ".up_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".up_proj"],
                n_scale_bits = 8,
                n_bits = 8,
            )
            m.down_proj = W8AX_LinearStatic.from_float(
                m.down_proj,
                decoder_scales["model." + name + ".down_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".down_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name = "model." + name + ".down_proj",
            )
        elif isinstance(m, Qwen2VLSdpaAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            # m.q_proj = W8AX_LinearStatic.from_float(
            #     m.q_proj,
            #     decoder_scales["model." + name + ".q_proj"],
            #     weight_quant_type=weight_quant,
            #     clip_top=layer_clip["model." + name + ".q_proj"],
            #     n_scale_bits = 16,
            #     n_bits = 16,
            #     name= "model." + name + ".q_proj",
            # )
            # m.k_proj = W8AX_LinearStatic.from_float(
            #     m.k_proj,
            #     decoder_scales["model." + name + ".k_proj"],
            #     weight_quant_type=weight_quant,
            #     clip_top=layer_clip["model." + name + ".k_proj"],
            #     n_scale_bits = 16,
            #     n_bits = 16,
            #     name= "model." + name + ".k_proj",
            # )
            # m.v_proj = W8AX_LinearStatic.from_float(
            #     m.v_proj,
            #     decoder_scales["model." + name + ".v_proj"],
            #     weight_quant_type=weight_quant,
            #     clip_top=layer_clip["model." + name + ".v_proj"],
            #     n_scale_bits = 16,
            #     n_bits = 16,
            #     name= "model." + name + ".v_proj",
            # )
            
            m.o_proj = W8AX_LinearStatic.from_float(
                m.o_proj,
                decoder_scales["model." + name + ".o_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".o_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name= "model." + name + ".o_proj",
            )
    return model


def quantize_qwen2vl_qkvnobias_like(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=False,
    layer_clip=None,
    quantize_ViT=False,
    nbits_dict=None,
):
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VLSdpaAttention,
        Qwen2MLP,
        VisionMlp,
        VisionSdpaAttention,
    )
    
    assert layer_clip is not None, "layer_clip should not be None"
    nbits_dict = nbits_dict if nbits_dict is not None else {}
    
    if quantize_ViT:
        for name, m in model.visual.named_modules():
            if isinstance(m, VisionMlp):
                print(f"quantize: {name}.fc1/fc2")
                m.fc1 = W8AX_Nobias_LinearStatic.from_float(
                    m.fc1,
                    decoder_scales["visual." + name + ".fc1"],
                    weight_quant_type=weight_quant,
                    clip_top=layer_clip["visual." + name + ".fc1"],
                    n_scale_bits = 8,
                    n_bits = 8,
                )
                print(f"{"visual." + name + ".fc1"}: {decoder_scales['visual.' + name + '.fc1']} {layer_clip['visual.' + name + '.fc1']}")
                m.fc2 = W8AX_Nobias_LinearStatic.from_float(
                    m.fc2,
                    decoder_scales["visual." + name + ".fc2"],
                    weight_quant_type=weight_quant,
                    clip_top=layer_clip["visual." + name + ".fc2"],
                    n_scale_bits = 8,
                    n_bits = 8,
                )
                print(f"{"visual." + name + ".fc2"}: {decoder_scales['visual.' + name + '.fc2']} {layer_clip['visual.' + name + '.fc2']}")
            
            if isinstance(m, VisionSdpaAttention):
                print(f"quantize: {name}.qkv/proj")
                m.qkv = W8AX_Nobias_LinearStatic.from_float(
                    m.qkv,
                    decoder_scales["visual." + name + ".qkv"],
                    weight_quant_type=weight_quant,
                    clip_top=layer_clip["visual." + name + ".qkv"],
                    n_scale_bits = 16,
                    n_bits = 16,
                    name= "visual." + name + ".qkv",
                )
                print(f"{"visual." + name + ".qkv"}: {decoder_scales['visual.' + name + '.qkv']} {layer_clip['visual.' + name + '.qkv']}")
                m.proj = W8AX_Nobias_LinearStatic.from_float(
                    m.proj,
                    decoder_scales["visual." + name + ".proj"],
                    weight_quant_type=weight_quant,
                    clip_top=layer_clip["visual." + name + ".proj"],
                    n_scale_bits = 8,
                    n_bits = 8,
                    name= "visual." + name + ".proj",
                )
                print(f"{"visual." + name + ".proj"}: {decoder_scales['visual.' + name + '.proj']} {layer_clip['visual.' + name + '.proj']}")

    for name, m in model.model.named_modules():
        if isinstance(m, Qwen2MLP):
            print(f"{'model.' + name + '.down_proj'}: nbits: {nbits_dict.get('model.' + name + '.down_proj', 8)}")
            print(f"quantize: {name}.gate_proj/up_proj/down_proj")
            m.gate_proj = W8AX_LinearStatic.from_float(
                m.gate_proj,
                decoder_scales["model." + name + ".gate_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".gate_proj"],
                n_scale_bits = 8,
                n_bits = 8,
            )
            print(f"{"model." + name + ".gate_proj"}: {decoder_scales['model.' + name + '.gate_proj']} {layer_clip['model.' + name + '.gate_proj']}")
            m.up_proj = W8AX_LinearStatic.from_float(
                m.up_proj,
                decoder_scales["model." + name + ".up_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".up_proj"],
                n_scale_bits = 8,
                n_bits = 8,
            )
            print(f"{"model." + name + ".up_proj"}: {decoder_scales['model.' + name + '.up_proj']} {layer_clip['model.' + name + '.up_proj']}")
            m.down_proj = W8AX_LinearStatic.from_float(
                m.down_proj,
                decoder_scales["model." + name + ".down_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".down_proj"],
                n_scale_bits = nbits_dict.get("model." + name + ".down_proj", 8),
                n_bits = nbits_dict.get("model." + name + ".down_proj", 8),
                name = "model." + name + ".down_proj",
            )
            print(f"{"model." + name + ".down_proj"}: {decoder_scales['model.' + name + '.down_proj']} {layer_clip['model.' + name + '.down_proj']}")
        elif isinstance(m, Qwen2VLSdpaAttention):
            print(f"quantize: {name}.q_proj/k_proj/v_proj/o_proj")
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8AX_Nobias_LinearStatic.from_float(
                m.q_proj,
                decoder_scales["model." + name + ".q_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".q_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name= "model." + name + ".q_proj",
            )
            print(f"{"model." + name + ".q_proj"}: {decoder_scales['model.' + name + '.q_proj']} {layer_clip['model.' + name + '.q_proj']}")
            m.k_proj = W8AX_Nobias_LinearStatic.from_float(
                m.k_proj,
                decoder_scales["model." + name + ".k_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".k_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name= "model." + name + ".k_proj",
            )
            print(f"{"model." + name + ".k_proj"}: {decoder_scales['model.' + name + '.k_proj']} {layer_clip['model.' + name + '.k_proj']}")
            m.v_proj = W8AX_Nobias_LinearStatic.from_float(
                m.v_proj,
                decoder_scales["model." + name + ".v_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".v_proj"],
                n_scale_bits = 16,
                n_bits = 16,
                name= "model." + name + ".v_proj",
            )
            print(f"{"model." + name + ".v_proj"}: {decoder_scales['model.' + name + '.v_proj']} {layer_clip['model.' + name + '.v_proj']}")
            m.o_proj = W8AX_LinearStatic.from_float(
                m.o_proj,
                decoder_scales["model." + name + ".o_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".o_proj"],
                n_scale_bits = 8,
                n_bits = 8,
                name= "model." + name + ".o_proj",
            )
            print(f"{"model." + name + ".o_proj"}: {decoder_scales['model.' + name + '.o_proj']} {layer_clip['model.' + name + '.o_proj']}")
    return model

def quantize_gemma_like(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=False,
    layer_clip={},
):
    from transformers.models.gemma.modeling_gemma import (
        GemmaSdpaAttention,
        GemmaMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, GemmaMLP):
            m.gate_proj = W8A8LinearStatic.from_float(
                m.gate_proj,
                decoder_scales["model." + name + ".gate_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".gate_proj"],
            )
            m.up_proj = W8A8LinearStatic.from_float(
                m.up_proj,
                decoder_scales["model." + name + ".up_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".up_proj"],
            )
            m.down_proj = W8A8LinearStatic.from_float(
                m.down_proj,
                decoder_scales["model." + name + ".down_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".down_proj"],
            )
        elif isinstance(m, GemmaSdpaAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales["model." + name + ".q_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".q_proj"],
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales["model." + name + ".k_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".k_proj"],
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales["model." + name + ".v_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".v_proj"],
            )
            m.o_proj = W8A8LinearStatic.from_float(
                m.o_proj,
                decoder_scales["model." + name + ".o_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".o_proj"],
            )
    return model


def quantize_phi_like(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=False,
    layer_clip={},
):
    from transformers.models.phi.modeling_phi import (
        PhiAttention,
        PhiMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, PhiMLP):
            m.fc1 = W8A8LinearStatic.from_float(
                m.fc1,
                decoder_scales["model." + name + ".fc1"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".fc1"],
            )
            m.fc2 = W8A8LinearStatic.from_float(
                m.fc2,
                decoder_scales["model." + name + ".fc2"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".fc2"],
            )

        elif isinstance(m, PhiAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales["model." + name + ".q_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".q_proj"],
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales["model." + name + ".k_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".k_proj"],
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales["model." + name + ".v_proj"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".v_proj"],
            )
            m.dense = W8A8LinearStatic.from_float(
                m.dense,
                decoder_scales["model." + name + ".dense"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip["model." + name + ".dense"],
            )
    return model


def quantize_mixtral(
    model, weight_quant="per_channel", act_quant="per_token", quantize_bmm_input=False
):
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
        MixtralBLockSparseTop2MLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, MixtralBLockSparseTop2MLP):
            m.w1 = W8A8LinearStatic.from_float(
                m.w1, weight_quant_type=weight_quant, act_quant=act_quant
            )
            m.w2 = W8A8LinearStatic.from_float(
                m.w2, weight_quant_type=weight_quant, act_quant=act_quant
            )
            m.w3 = W8A8LinearStatic.from_float(
                m.w3, weight_quant_type=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                weight_quant_type=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                weight_quant_type=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                weight_quant_type=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8LinearStatic.from_float(
                m.o_proj, weight_quant_type=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W8A8LinearStatic.from_float(
                m.gate, weight_quant_type=weight_quant, act_quant=act_quant
            )
    return model


def quantize_falcon_like(
    model,
    decoder_scales,
    weight_quant="per_tensor",
    act_quant="per_tensor",
    quantize_bmm_input=False,
    layer_clip={},
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            m.dense_h_to_4h = W8A8LinearStatic.from_float(
                m.dense_h_to_4h,
                decoder_scales[name + ".dense_h_to_4h"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip[name + ".dense_h_to_4h"],
            )
            m.dense_4h_to_h = W8A8LinearStatic.from_float(
                m.dense_4h_to_h,
                decoder_scales[name + ".dense_4h_to_h"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip[name + ".dense_4h_to_h"],
            )
        elif isinstance(m, FalconAttention):
            # Here we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.query_key_value = W8A8LinearStatic.from_float(
                m.query_key_value,
                decoder_scales[name + ".query_key_value"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip[name + ".query_key_value"],
            )
            m.dense = W8A8LinearStatic.from_float(
                m.dense,
                decoder_scales[name + ".dense"],
                weight_quant_type=weight_quant,
                clip_top=layer_clip[name + ".dense"],
            )
    return model