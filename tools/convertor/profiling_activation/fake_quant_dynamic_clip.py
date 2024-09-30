import torch
from torch import nn
from functools import partial


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8):
    scales = w.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_().mul_(scales)
    return w

@torch.no_grad()
def quantize_weight_scale_per_tensor_absmax(w, n_bits=8):
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    return scales

@torch.no_grad()
def quantize_activation_per_tensor_static_input(t, scale=1, n_bits=8, clip_top=False):
    scale = scale.clone().to(t.device)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    scale.clamp_(min=1e-5).div_(q_max)
    scale = scale * 100000
    scale = scale.round() / 100000
    t = t.div(scale).round()
    if clip_top:
        t = t.clamp(-128.0, 127.0)
    torch.set_printoptions(profile="full")
    print('linear quantized int8 inputs')
    print(t.shape)
    print(t)
    return t

@torch.no_grad()
def quantize_weight_per_tensor_static_input(t, scale=1, n_bits=8, clip_top=False):
    scale = scale.clone().to(t.device)
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    t = t.div(scale).round()

    return t

@torch.no_grad()
def quantize_activation_per_tensor_static_output(t, scale=1, n_bits=8, clip_top=False):
    scale = scale.clone().to(t.device)

    t_shape = t.shape
    t.view(-1, t_shape[-1])
    q_max = 2 ** (n_bits - 1) - 1
    scale.clamp_(min=1e-5).div_(q_max)


    print(scale)
    t = t.div(torch.floor(1.0/scale))

    torch.set_printoptions(profile="full")
    print('linear float outputs')
    print(t.shape)
    print(t)

    t = t.round()
    if clip_top:
        t = t.clamp(-128.0, 127.0)

    torch.set_printoptions(profile="full")
    print('linear quantized int8 outputs')
    print(t.shape)
    print(t)

    t = t.div(scale)
    return t

@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8):
    t_shape = t.shape
    t.view(-1, t_shape[-1])
    scales = t.abs().max(dim=-1, keepdim=True)[0]
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    t.div_(scales).round_().mul_(scales)
    print(t)
    return t


@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8):
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
        clip_top=False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_scale = torch.tensor(input_scale)
        self.output_scale = torch.tensor(output_scale)

        self.weight_scale = None

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
        if clip_top['input']:
            print('input clip top!!')
        if clip_top['output']:
            print('output clip top!!')
        self.act_quant_input = partial(quantize_activation_per_tensor_static_input, n_bits=8, clip_top=clip_top['input'])

        self.output_quant = partial(quantize_activation_per_tensor_static_output, n_bits=8, clip_top=clip_top['output'])

        self.weight_quant = partial(quantize_weight_per_tensor_static_input, n_bits=8, clip_top=False)

    def to(self, *args, **kwargs):
        super(W8A8LinearStatic, self).to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        if self.bias is not None:
            self.bias = self.bias.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(self, x):
        torch.set_printoptions(profile="full")
        print('linear float inputs')
        print(x.shape)
        print(x)

        q_x = self.act_quant_input(x, scale=self.input_scale)
        w = self.weight_quant(self.weight, scale=self.weight_scale)

        torch.set_printoptions(profile="full")
        print('linear quantized float inputs')
        print(q_x.shape)
        print(q_x)
        
        y = torch.functional.F.linear(q_x, w, self.bias)

        torch.set_printoptions(profile="full")
        print('linear quantized int32 outputs')
        print(y.shape)
        print(y)

        o_scale =  (torch.round(self.input_scale * 100000) / 100000 ) * self.weight_scale / (torch.round(self.output_scale * 100000) / 100000 )
        q_y = self.output_quant(y, scale=o_scale)

        torch.set_printoptions(profile="full")
        print('linear quantized float outputs')
        print(q_y.shape)
        print(q_y)

        return q_y #+ yplus

    @staticmethod
    def from_float(
        module, scales, weight_quant='per_tensor', clip_top=False
    ):
        assert isinstance(module, torch.nn.Linear)

        new_module = W8A8LinearStatic(
            module.in_features,
            module.out_features,
            bias = module.bias is not None,
            input_scale=scales['input'],
            output_scale=scales['output'],
            clip_top=clip_top     
        )
        if weight_quant == "per_channel":
            new_module.weight = quantize_weight_per_channel_absmax(
                module.weight, n_bits=8
            )  # use 8-bit integer for weight
        elif weight_quant == "per_tensor":
            new_module.weight = quantize_weight_per_tensor_absmax(
                module.weight, n_bits=8
            )
        else:
            raise ValueError(f"Invalid weight_quant: {weight_quant}")
        new_module.weight_quant_name = weight_quant
        if module.bias is not None:
            new_module.bias = quantize_weight_per_tensor_absmax(
                module.bias, n_bits=8
            )
        
        new_module.weight_scale = quantize_weight_scale_per_tensor_absmax(
                module.weight, n_bits=8
            )
        return new_module

    def __repr__(self):
        return f"W8A8LinearStatic({self.in_features}, {self.out_features}, bias={self.bias is not None}, weight_quant={self.weight_quant_name}, input_scale={self.input_scale.item()}, output_scale={self.output_scale.item()}, clip_top={self.act_quant_input.keywords['clip_top']})"


def quantize_opt(
    model, decoder_scales, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=True, layer_clip = {}
):
    from transformers.models.opt.modeling_opt import (
        OPTAttention,
        OPTDecoderLayer,
    )

    for name, m in model.model.named_modules():

        if isinstance(m, OPTDecoderLayer):
            print(name)
            print(decoder_scales['model.' + name + '.fc1'])
            m.fc1 = W8A8LinearStatic.from_float(
                m.fc1, decoder_scales['model.' + name + '.fc1'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.fc1']
            )
            m.fc2 = W8A8LinearStatic.from_float(
                m.fc2, decoder_scales['model.' + name + '.fc2'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.fc2']
            )
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales['model.' + name + '.q_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.q_proj']
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales['model.' + name + '.k_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.k_proj']
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales['model.' + name + '.v_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.v_proj']
            )
            m.out_proj = W8A8LinearStatic.from_float(
                m.out_proj, decoder_scales['model.' + name + '.out_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.out_proj']
            )
    return model


def quantize_llama_like(
    model, decoder_scales, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=False, layer_clip = {}
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
            print(name)
            m.gate_proj = W8A8LinearStatic.from_float(
                m.gate_proj, decoder_scales['model.' + name + '.gate_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.gate_proj']
            )
            m.up_proj = W8A8LinearStatic.from_float(
                m.up_proj, decoder_scales['model.' + name + '.up_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.up_proj']
            )
            m.down_proj = W8A8LinearStatic.from_float(
                m.down_proj, decoder_scales['model.' + name + '.down_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.down_proj']
            )
        elif isinstance(m, (LlamaAttention, MistralAttention)):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales['model.' + name + '.q_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.q_proj']
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales['model.' + name + '.k_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.k_proj']
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales['model.' + name + '.v_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.v_proj']
            )
            m.o_proj = W8A8LinearStatic.from_float(
                m.o_proj, decoder_scales['model.' + name + '.o_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.o_proj']
            )
    return model


def quantize_qwen2_like(
    model, decoder_scales, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=False, layer_clip = {}
):
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention,
        Qwen2MLP,
    )
        
    for name, m in model.model.named_modules():
        if isinstance(m, Qwen2MLP):
            m.gate_proj = W8A8LinearStatic.from_float(
                m.gate_proj, decoder_scales['model.' + name + '.gate_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.gate_proj']
            )
            m.up_proj = W8A8LinearStatic.from_float(
                m.up_proj, decoder_scales['model.' + name + '.up_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.up_proj']
            )
            m.down_proj = W8A8LinearStatic.from_float(
                m.down_proj, decoder_scales['model.' + name + '.down_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.down_proj']
            )
        elif isinstance(m, Qwen2Attention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            print(name)
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales['model.' + name + '.q_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.q_proj']
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales['model.' + name + '.k_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.k_proj']
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales['model.' + name + '.v_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.v_proj']
            )
            m.o_proj = W8A8LinearStatic.from_float(
                m.o_proj, decoder_scales['model.' + name + '.o_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.o_proj']
            )
    return model


def quantize_gemma_like(
    model, decoder_scales, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=False, layer_clip = {}
):
    from transformers.models.gemma.modeling_gemma import (
        GemmaSdpaAttention,
        GemmaMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, GemmaMLP):
            m.gate_proj = W8A8LinearStatic.from_float(
                m.gate_proj, decoder_scales['model.' + name + '.gate_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.gate_proj']
            )
            m.up_proj = W8A8LinearStatic.from_float(
                m.up_proj, decoder_scales['model.' + name + '.up_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.up_proj']
            )
            m.down_proj = W8A8LinearStatic.from_float(
                m.down_proj, decoder_scales['model.' + name + '.down_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.down_proj']
            )
        elif isinstance(m, GemmaSdpaAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales['model.' + name + '.q_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.q_proj']
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales['model.' + name + '.k_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.k_proj']
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales['model.' + name + '.v_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.v_proj']
            )
            m.o_proj = W8A8LinearStatic.from_float(
                m.o_proj, decoder_scales['model.' + name + '.o_proj'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.o_proj']
            )
    return model



def quantize_phi_like(
    model, decoder_scales, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=False, layer_clip = {}
):
    from transformers.models.phi.modeling_phi import (
        PhiAttention,
        PhiMLP,
    )

    for name, m in model.model.named_modules():
        if isinstance(m, PhiMLP):
            m.fc1 = W8A8LinearStatic.from_float(
                m.fc1, decoder_scales['model.' + name + '.fc1'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.fc1']
            )
            m.fc2 = W8A8LinearStatic.from_float(
                m.fc2, decoder_scales['model.' + name + '.fc2'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.fc2']
            )

        elif isinstance(m, PhiAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                decoder_scales['model.' + name + '.q_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.q_proj']
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                decoder_scales['model.' + name + '.k_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.k_proj']
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                decoder_scales['model.' + name + '.v_proj'],
                weight_quant=weight_quant,
                clip_top=layer_clip['model.' + name + '.v_proj']
            )
            m.dense = W8A8LinearStatic.from_float(
                m.dense, decoder_scales['model.' + name + '.dense'], weight_quant=weight_quant, clip_top=layer_clip['model.' + name + '.dense']
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
                m.w1, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w2 = W8A8LinearStatic.from_float(
                m.w2, weight_quant=weight_quant, act_quant=act_quant
            )
            m.w3 = W8A8LinearStatic.from_float(
                m.w3, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8LinearStatic.from_float(
                m.q_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.k_proj = W8A8LinearStatic.from_float(
                m.k_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.v_proj = W8A8LinearStatic.from_float(
                m.v_proj,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quantize_output=quantize_bmm_input,
            )
            m.o_proj = W8A8LinearStatic.from_float(
                m.o_proj, weight_quant=weight_quant, act_quant=act_quant
            )
        elif isinstance(m, MixtralSparseMoeBlock):
            m.gate = W8A8LinearStatic.from_float(
                m.gate, weight_quant=weight_quant, act_quant=act_quant
            )
    return model


def quantize_falcon_like(
    model, decoder_scales, weight_quant="per_tensor", act_quant="per_tensor", quantize_bmm_input=False, layer_clip = {}
):
    from transformers.models.falcon.modeling_falcon import (
        FalconAttention,
        FalconMLP,
    )

    for name, m in model.named_modules():
        if isinstance(m, FalconMLP):
            print(1, name, layer_clip[name + '.dense_h_to_4h'])
            m.dense_h_to_4h = W8A8LinearStatic.from_float(
                m.dense_h_to_4h, decoder_scales[name + '.dense_h_to_4h'], weight_quant=weight_quant, clip_top=layer_clip[name + '.dense_h_to_4h']
            )
            m.dense_4h_to_h = W8A8LinearStatic.from_float(
                m.dense_4h_to_h, decoder_scales[name + '.dense_4h_to_h'], weight_quant=weight_quant, clip_top=layer_clip[name + '.dense_4h_to_h']
            )
        elif isinstance(m, FalconAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            print(2, name, layer_clip[name + '.query_key_value'])
            m.query_key_value = W8A8LinearStatic.from_float(
                m.query_key_value,
                decoder_scales[name + '.query_key_value'],
                weight_quant=weight_quant,
                clip_top=layer_clip[name + '.query_key_value']
            )
            m.dense = W8A8LinearStatic.from_float(
                m.dense, decoder_scales[ name + '.dense'], weight_quant=weight_quant, clip_top=layer_clip[name + '.dense']
            )
    return model
