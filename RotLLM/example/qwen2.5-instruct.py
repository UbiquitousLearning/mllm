# -*- coding: utf-8 -*-

# This file shows how to rotate Qwen models using the package rotate.

import rotate
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/Qwen2.5-1.5B-Instruct" # specify the path to the model

device = "cuda:7" # specify the device to use
dtype = "float32" # specify the data type

def chat(tokenizer, model, prompt, max_new_tokens=1024):
    chats = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    prompt_text = tokenizer.apply_chat_template(chats, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    return response

if __name__ == "__main__":
    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=dtype)
    model.eval()

    prompt = "write me a binary search in C"
    response = chat(tokenizer, model, prompt)
    print(response)
    
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
    
    # test the rotated model
    print("--------------------------------------")
    response = chat(tokenizer, model, prompt)
    print(response)
    
    
    # now you can save the rotated model
    
    model.save_pretrained(model_path + "_rotated")
    tokenizer.save_pretrained(model_path + "_rotated")
    print(f"Rotated model saved to {model_path}_rotated")
    