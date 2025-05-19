"""
This file is a simulation of the inference process of a model that has been quantized using the quantization functions in the `quantization_simulation.py` file. 
The model is loaded from the path specified in the `model_name` argument, and the activation scales are loaded from the path specified in the `scale_file` argument. The `t01m_clip_threshold` argument specifies the threshold for clipping the activations. 
The model is quantized using the specified `model_type` argument, which determines the quantization function to be used. The quantized model is then used to generate an example based on the provided prompt.
"""

import argparse
import json

import torch
from utils.get_input_output_scales import get_clip_and_scale
from utils.quantization_simulation import (
    quantize_qwen2_like,
    quantize_llama_like,
    quantize_gemma_like,
    quantize_opt,
    quantize_phi_like,
    quantize_mixtral,
    quantize_falcon_like,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# prompt = """"Large Language Models (LLMs) are advanced artificial intelligence systems designed to understand and generate human-like text. These models are trained on vast amounts of data, enabling them to perform a wide range of tasks, from answering questions and summarizing text to generating creative content and engaging in conversational dialogue. LLMs like GPT-3 and GPT-4, developed by OpenAI, have set new benchmarks in natural language processing by leveraging deep learning architectures, particularly transformer models, which excel at capturing context and relationships within text. The scalability and versatility of LLMs make them invaluable tools for applications in education, customer service, content creation, and more. However, their deployment also raises ethical considerations, including issues of bias, misinformation, and the potential for misuse. As the field continues to evolve, ongoing research and responsible deployment strategies are essential to harnessing the full potential of these powerful AI systems while mitigating their risks."
# Generate a title based on the above text.
# """

prompt = "Give me a short introduction to large language model."

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument(
        "--model_type",
        choices=["llama", "qwen1", "qwen2", "gemma", "phi", "opt", "mixtral", "falcon"],
        default="llama",
    )
    parser.add_argument("--scale_file", type=argparse.FileType("r"))
    parser.add_argument("--t01m_clip_threshold", type=int, default=152)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    act_dict = json.load(open(args.scale_file.name))

    act_scales, clip_top, return_dict = get_clip_and_scale(act_dict, args.t01m_clip_threshold)

    print(f"clip input num: {return_dict['clip_input_num']}")
    print(f"clip output num: {return_dict['clip_output_num']}")
    print(f"no clip input num: {return_dict['no_clip_input_num']}")
    for i in return_dict["no_clip_input_name"]:
        print(f"no clip input: {i}")
    print(f"no clip output num: {return_dict['no_clip_output_num']}")
    for i in return_dict["no_clip_output_name"]:
        print(f"no clip output: {i}")

    if args.model_type == "llama":
        q_model = quantize_llama_like(model, act_scales, layer_clip=clip_top)
    elif args.model_type == "qwen2" or args.model_type == "qwen1":
        q_model = quantize_qwen2_like(model, act_scales, layer_clip=clip_top)
    elif args.model_type == "gemma":
        q_model = quantize_gemma_like(model, act_scales, layer_clip=clip_top)
    elif args.model_type == "phi":
        q_model = quantize_phi_like(model, act_scales, layer_clip=clip_top)
    elif args.model_type == "opt":
        q_model = quantize_opt(model, act_scales, layer_clip=clip_top)
    elif args.model_type == "mixtral":
        q_model = quantize_mixtral(model, act_scales, layer_clip=clip_top)
    elif args.model_type == "falcon":
        q_model = quantize_falcon_like(model, act_scales, layer_clip=clip_top)
    else:
        print("Model type not supported")
        exit(1)

    # use q_model to generate an example
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = q_model.generate(
            **input_ids, max_length=100, do_sample=False, top_p=None, top_k=None
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))
