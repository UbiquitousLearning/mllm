import torch
from torch import nn
from typing import Union
from transformers import Qwen2ForCausalLM
from ..common import RotateOperationRegistry
from ..common import AutoOperation

from ..common import NormLinearIterator
try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel, Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.modeling_qwen2_vl import PatchEmbed
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLAttention
    from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention, VisionFlashAttention2, VisionSdpaAttention
    HAS_QWEN2_VL = True
except ImportError:
    HAS_QWEN2_VL = False
    class Qwen2VLForConditionalGeneration:
        pass
    class Qwen2VisionTransformerPretrainedModel:
        pass
    class PatchEmbed:
        pass
    class Qwen2VLAttention:
        pass
    class VisionAttention:
        pass
    class VisionFlashAttention2:
        pass
    class VisionSdpaAttention:
        pass
    
    
   
@NormLinearIterator.register_iterator
class Qwen2NormLinearIterator(NormLinearIterator):
    def __init__(self, model: Qwen2ForCausalLM):
        super().__init__()
        self.model = model
        
    def __iter__(self):
        for layer in self.model.model.layers:
            yield layer, "input_layernorm", [
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
            ]
            yield layer, "post_attention_layernorm", [
                layer.mlp.up_proj,
                layer.mlp.gate_proj,
            ]
        yield self.model.model, "norm", [self.model.lm_head]
        
    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, Qwen2ForCausalLM) or isinstance(model, Qwen2VLForConditionalGeneration)
    

@NormLinearIterator.register_iterator
class Qwen2ViTNormLinearIterator(NormLinearIterator):
    def __init__(self, model: Qwen2VisionTransformerPretrainedModel):
        super().__init__()
        self.model = model
        
    def __iter__(self):
        for layer in self.model.blocks:
            yield layer, "norm1", [layer.attn.qkv]
            yield layer, "norm2", [layer.mlp.fc1]
        yield self.model.merger, "ln_q", [self.model.merger.mlp[0]]
    
    @classmethod
    def supports_model(cls, model: nn.Module) -> bool:
        return isinstance(model, Qwen2VisionTransformerPretrainedModel)


@AutoOperation.register_operation("rotate_output", PatchEmbed)
def op_rotate_patch_embed_output(
    patch_embed: PatchEmbed,
    R: torch.Tensor):
    linear = patch_embed.proj
    assert R.shape[0] == R.shape[1], "R should be a square matrix"
    assert R.shape[0] == linear.weight.shape[0], "R should be same size as output dim of linear layer"
    dtype = linear.weight.dtype
    shape = linear.weight.shape
    R_device = R.device
    w_device = linear.weight.device
    W_ = linear.weight.data.to(device=R_device, dtype=torch.float64).view(R.shape[0], -1)
    # note that the W_ in linear is transpose of W
    linear.weight.data = (R.T.to(torch.float64) @ W_).to(device=w_device, dtype=dtype).reshape(shape)
    

@AutoOperation.register_operation("center_output", PatchEmbed)
def op_center_patch_embed_output(patch_embed: PatchEmbed):
    linear = patch_embed.proj
    dtype = linear.weight.dtype
    W_ = linear.weight.data.to(dtype=torch.float64).view(linear.weight.shape[0], -1)
    # note that the W_ in linear is transpose of W
    # center echo columns of W equivalent to centering the rows of W_
    W_mean = W_.mean(dim=0, keepdim=True)
    W_centered = W_ - W_mean
    linear.weight.data = W_centered.to(dtype=dtype).reshape(linear.weight.shape)

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

@AutoOperation.register_operation("rotate_attn_v", Qwen2Attention)
@AutoOperation.register_operation("rotate_attn_v", Qwen2VLAttention)
def op_rotate_attn_v_for_LM(
    attn: Union[Qwen2Attention, Qwen2VLAttention],
    R_v: torch.Tensor):
    """
    rotate the v (one of the inputs of attention) by a rotation matrix R_v 
    and rotate v back before W_o
    """
    config = attn.config
    num_qo_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    
    # rotate v in attention
    # i.e. rotate the output of W_v
    # note that the output is something like [v_1, v_2, ..., v_{num_heads}]
    # where v_i is a head_dim vector
    # so we need to rotate each head
    # results should be something like [v_1R_v, v_2R_v, ..., v_{num_heads}R_v]
    # this is equal to [v_1, v_2, ..., v_{num_heads}] @ diag(R_v, R_v, ..., R_v) (num_heads times)
    # so we need to rotate the output of W_v by diag(R_v, R_v, ..., R_v)
    R_v_rot = torch.block_diag(*([R_v] * num_kv_heads))
    # rotate_linear_output([attn.v_proj], R_v_rot)
    AutoOperation.rotate_output(attn.v_proj, R_v_rot)
    
    # then we need to rotate back the input of W_o
    # since o_i is linear combination of v_i
    # we can rotate the o_i by R_v^T to get back the original o_i
    # rotate_linear_input([attn.o_proj], torch.block_diag(*([R_v] * num_qo_heads)).T)
    AutoOperation.rotate_input(attn.o_proj, torch.block_diag(*([R_v] * num_qo_heads)).T)




@AutoOperation.register_operation("rotate_attn_v", VisionAttention)
@AutoOperation.register_operation("rotate_attn_v", VisionFlashAttention2)
@AutoOperation.register_operation("rotate_attn_v", VisionSdpaAttention)
def op_rotate_attn_v_for_ViT(
    attn: Union[VisionAttention, VisionFlashAttention2, VisionSdpaAttention],
    R_v: torch.Tensor):
    """
    rotate the v (one of the inputs of attention) by a rotation matrix R_v 
    and rotate v back before W_o
    """
    num_heads = attn.num_heads
    dim = attn.proj.weight.shape[0]
    head_dim = dim // num_heads
    
    
    # shape of qkv.weight: [3 * dim, dim]
    q_proj, k_proj, v_proj = attn.qkv.weight.view(3, dim, dim).unbind(0) # now shape of v_proj is [dim, dim] (out_dim, in_dim)
    q_bias, k_bias, v_bias = attn.qkv.bias.view(3, dim).unbind(0)
    
    # v_proj: [dim, dim] can be view as [num_heads * head_dim, dim]
    # view it as [num_heads, head_dim, dim]
    dtype = v_proj.dtype
    device = v_proj.device
    R_device = R_v.device
    v_proj = v_proj.view(num_heads, head_dim, dim).to(device=R_device, dtype=torch.float64)
    v_proj = (R_v.T.unsqueeze(0).to(torch.float64) @ v_proj).to(device=device, dtype=dtype)
    v_proj = v_proj.view(dim, dim) # change it back to the original shape
    
    # rotate v_bias
    # v_bias: [dim]
    # which can be view as [num_heads, head_dim]
    v_bias = v_bias.view(num_heads, head_dim).to(dtype=torch.float64, device=R_device)
    v_bias = (v_bias @ R_v.to(torch.float64)).to(dtype=dtype, device=device).view(-1)
    
    # stack to get the original qkv back
    qkv = torch.stack([q_proj, k_proj, v_proj], dim=0)
    qkv = qkv.view(3 * dim, dim)
    qkv = qkv.to(device=device, dtype=dtype)
    attn.qkv.weight.data = qkv
    attn.qkv.bias.data = torch.cat([q_bias, k_bias, v_bias], dim=0)
    
    # rotate the output of W_o
    AutoOperation.rotate_input(attn.proj, torch.block_diag(*([R_v] * num_heads)).T)



def untie_word_embeddings(model):
    if model.config.tie_word_embeddings:
        # Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
        # this is because the weight of RMSNorm will be merge into lm_head
        # and this weight will not be merged into the embeddings
        # making the weights of lm_head and embed_tokens not the same
        print("tie word embeddings, clone lm_head from embed_tokens")
        model.config.tie_word_embeddings = False

        # create a new weight for lm_head
        new_weight = torch.empty_like(model.model.embed_tokens.weight)
        new_weight.copy_(model.model.embed_tokens.weight)

        # copy from model.model.embed_tokens.weight
        model.lm_head.weight = nn.Parameter(new_weight)
        new_weight = torch.empty_like(model.model.embed_tokens.weight)
        new_weight.copy_(model.model.embed_tokens.weight)

        # assign the new weight to lm_head
        model.lm_head.weight = nn.Parameter(new_weight)

        # ensure that the ptr of weight of lm_head is not the same as ptr of the weight of embed_tokens
        assert model.model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()
    

@torch.inference_mode()
def rotate_model(model: Union[Qwen2ForCausalLM, Qwen2VLForConditionalGeneration],
                 R: torch.Tensor,
                 R_v: list[torch.Tensor] = None):
    config = model.config
    dim = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = dim // num_heads
    num_layers = config.num_hidden_layers

    assert R.shape == (dim, dim), f"Rotation matrix shape {R.shape} does not match model dimension {dim}"

    if isinstance(R_v, torch.Tensor):
        # R_v is a single rotation matrix
        assert R_v.shape == (head_dim, head_dim), f"Rotation matrix shape {R_v.shape} does not match model dimension {dim}"
        R_v = [R_v for _ in range(num_layers)]

    assert R_v is None or len(R_v) == num_layers, f"number of rotation matrix {len(R_v)} does not match number of layers {num_layers}"
    assert all([R_v[i].shape == (head_dim, head_dim) for i in range(num_layers)]) if R_v is not None else True, f"Rotation matrix shape {R_v} does not match model dimension {dim}"

    # rotate embedding
    AutoOperation.rotate_output(model.model.embed_tokens, R)

    if isinstance(model, Qwen2VLForConditionalGeneration):
        # rotate the output of ViT
        merger = model.visual.merger
        AutoOperation.rotate_output(merger.mlp[2], R)


    for l, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        # reverse rotation for input of W_qkv
        AutoOperation.rotate_input(attn.q_proj, R.T)
        AutoOperation.rotate_input(attn.k_proj, R.T)
        AutoOperation.rotate_input(attn.v_proj, R.T)
        # rotate output of W_o
        AutoOperation.rotate_output(attn.o_proj, R)

        if R_v is not None:
            # rotate v in attention and rotate back before W_o
            AutoOperation.rotate_attn_v(attn, R_v[l])

        mlp = layer.mlp
        # reverse rotation for input of W_up and W_gate
        AutoOperation.rotate_input(mlp.up_proj, R.T)
        AutoOperation.rotate_input(mlp.gate_proj, R.T)
        # rotate output of W_down
        AutoOperation.rotate_output(mlp.down_proj, R)

    # reverse rotation for input of W_lm
    AutoOperation.rotate_input(model.lm_head, R.T)
    

def center_output_of_each_layer_for_qwen2_vit(model: Qwen2VisionTransformerPretrainedModel):
    """
    Center the output of each layer for Qwen2 ViT.
    """
    # extract the centering operation from LayerNorm to the previous layer
    # center the output of the patch embedding
    AutoOperation.center_output(model.patch_embed)
    
    for layer in model.blocks:
        attn = layer.attn
        # center the output of proj
        AutoOperation.center_output(attn.proj)
        
        mlp = layer.mlp
        # center the output of fc2
        AutoOperation.center_output(mlp.fc2)


@torch.inference_mode()
def rotate_qwen2_ViT(model: Qwen2VisionTransformerPretrainedModel,
                     R: torch.Tensor,
                     R_v: list[torch.Tensor] = None):
    config = model.config
    dim = config.embed_dim
    num_heads = config.num_heads
    head_dim = dim // num_heads
    num_layers = config.depth

    assert R.shape == (dim, dim), f"Rotation matrix shape {R.shape} does not match model dimension {dim}"

    if isinstance(R_v, torch.Tensor):
        # R_v is a single rotation matrix
        assert R_v.shape == (head_dim, head_dim), f"Rotation matrix shape {R_v.shape} does not match model dimension {dim}"
        R_v = [R_v for _ in range(num_layers)]

    assert R_v is None or len(R_v) == num_layers, f"number of rotation matrix {len(R_v)} does not match number of layers {num_layers}"
    assert all([R_v[i].shape == (head_dim, head_dim) for i in range(num_layers)]) if R_v is not None else True, f"Rotation matrix shape {R_v} does not match model dimension {dim}"

    # rotate embedding
    AutoOperation.rotate_output(model.patch_embed, R)

    for l, layer in enumerate(model.blocks):
        attn = layer.attn
        # reverse rotation for input of W_qkv
        AutoOperation.rotate_input(attn.qkv, R.T)
        # rotate output of W_o
        AutoOperation.rotate_output(attn.proj, R)

        if R_v is not None:
            # rotate v in attention and rotate back before W_o
            AutoOperation.rotate_attn_v(attn, R_v[l])

        mlp = layer.mlp
        # reverse rotation for input of W_up and W_gate
        AutoOperation.rotate_input(mlp.fc1, R.T)
        # rotate output of W_down
        AutoOperation.rotate_output(mlp.fc2, R)
    
    AutoOperation.rotate_input(model.merger.mlp[0], R.T)
        

@RotateOperationRegistry.register(Qwen2ForCausalLM)
@RotateOperationRegistry.register(Qwen2VLForConditionalGeneration)
def apply_untie_word_embeddings(model: Union[Qwen2ForCausalLM, Qwen2VLForConditionalGeneration], *args, **kwargs):
    """
    Untie the word embeddings of the model.
    """
    print("Untie word embeddings")
    untie_word_embeddings(model)

from ..rotation_utils import fuse_layer_norms

@RotateOperationRegistry.register(Qwen2ForCausalLM)
@RotateOperationRegistry.register(Qwen2VLForConditionalGeneration)
def apply_fuse_layer_norms(model: Union[Qwen2ForCausalLM, Qwen2VLForConditionalGeneration], *args, **kwargs):
    """
    Fuse the layer norms of the model.
    """
    print("Fuse layer norms")
    fuse_layer_norms(model)
    

@RotateOperationRegistry.register(Qwen2VisionTransformerPretrainedModel)
def apply_fuse_layer_norms_vit(model: Qwen2VisionTransformerPretrainedModel, *args, **kwargs):
    """
    Fuse the layer norms of the model.
    """
    print("Fuse layer norms for ViT of Qwen2")
    fuse_layer_norms(model, replace_ln=True)
    
@RotateOperationRegistry.register(Qwen2VisionTransformerPretrainedModel)
def apply_center_output_of_each_layer_for_qwen2_vit(model: Qwen2VisionTransformerPretrainedModel, *args, **kwargs):
    """
    Center the output of each layer for Qwen2 ViT.
    """
    print("Center output of each layer for Qwen2 ViT")
    center_output_of_each_layer_for_qwen2_vit(model)


@RotateOperationRegistry.register(Qwen2VisionTransformerPretrainedModel)
def apply_rotate_qwen2_ViT(model: Qwen2VisionTransformerPretrainedModel, *args, **kwargs):
    """
    Rotate the model.
    """
    print("Rotate ViT model")
    rotate_qwen2_ViT(model, *args, **kwargs)
    
    
@RotateOperationRegistry.register(Qwen2ForCausalLM)
@RotateOperationRegistry.register(Qwen2VLForConditionalGeneration)
def apply_rotate_model(model: Union[Qwen2ForCausalLM, Qwen2VLForConditionalGeneration], *args, **kwargs):
    """
    Rotate the model.
    """
    print("Rotate model")
    rotate_model(model, *args, **kwargs)
    
