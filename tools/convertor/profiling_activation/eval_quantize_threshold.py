"""
This file is for evaluating the accuracy of a model with differen thresholds for clipping the activations.
The threshold will control the number of activations that are clipped. If a layer is clipped, if means that this layer can be caculated using W8A8 with no fall back to FP32.
"""

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import multiprocessing
import json
from tqdm import tqdm
from datasets import load_dataset

from utils.quantization_simulation import (
    quantize_falcon_like,
    quantize_mixtral,
    quantize_qwen2_like,
    quantize_llama_like,
    quantize_gemma_like,
    quantize_opt,
    quantize_phi_like,
)
from utils.get_input_output_scales import get_clip_and_scale


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.dataset):
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc


def evaluate_model(model_name, act_dict, result_queue, t01m_thre):
    dataset = load_dataset("lambada", split="validation[:1000]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda:1")
    evaluator = Evaluator(dataset, tokenizer, "cuda:1")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda:1")

    act_scales, clip_top, return_dict = get_clip_and_scale(act_dict, t01m_thre)

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

    res = evaluator.evaluate(q_model)
    print(t01m_thre, res)

    return_dict["res"] = float(res)
    result_queue.put((t01m_thre, return_dict))


def get_all_actscale_result_parallel(model_name, act_dict):
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    for t01m_thre in [1, 2, 4, 8, 16, 24,  32, 64, 128, 152, 10000000]:
        p = multiprocessing.Process(
            target=evaluate_model, args=(model_name, act_dict, result_queue, t01m_thre)
        )
        p.start()
        p.join()

    results = {}
    while not result_queue.empty():
        key, value = result_queue.get()
        results[key] = value

    return results


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
    args = parser.parse_args()

    res_data = {}

    act_dict = json.load(open(args.scale_file.name))
    results = get_all_actscale_result_parallel(
        args.model_name, act_dict
    )
    res_data[args.model_name] = results

    with open("model_res_acc.json", "w") as f:
        json.dump(res_data, f, indent=4, ensure_ascii=False)
    print("write to model_res_acc.json")
