# -*- coding: utf-8 -*-

# This file shows how to rotate Qwen models using the package rotate.
import torch
import rotate
from transformers import Qwen2VLForConditionalGeneration

model_path = "path/to/ShowUI-2B" # specify the path to the model

device = "cuda:7" # specify the device to use
dtype = "float32" # specify the data type


if __name__ == "__main__":
    # load the model and tokenizer
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device,
    )
    
    # model info
    num_layers = model.config.num_hidden_layers
    dim = model.config.hidden_size
    qo_heads = model.config.num_attention_heads
    head_dim = dim // qo_heads
    
    # get randome hadamard rotation matrix
    R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
    R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard", device=device) for _ in range(num_layers)]
    # rotate the model using the rotation matrix
    # currently only supports Qwen2ForCausalLM and Qwen2VLForConditionalGeneration
    rotate.rotate_model(model, R, R_v)
    
    # if you want to also rotate the ViT of Qwen2VLForConditionalGeneration, you can run the following line:
    vit_dim = model.config.vision_config.embed_dim
    vit_heads = model.config.vision_config.num_heads
    vit_head_dim = vit_dim // vit_heads
    vit_layers = model.config.vision_config.depth
    R_vit = rotate.get_orthogonal_matrix(vit_dim, mode="hadamard", device=device)
    R_v_vit = [rotate.get_orthogonal_matrix(vit_head_dim, mode="hadamard", device=device) for _ in range(vit_layers)]
    rotate.rotate_model(model.visual, R_vit, R_v_vit)
    
    
    # now you can save the rotated model
    model.save_pretrained(model_path + "-lm-vit-rotated")

    