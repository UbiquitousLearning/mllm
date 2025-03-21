# NOTE:Need a HUGE amount of memory(>=100GB) and time(>=1 hour) to run this script
"""
This file is used to get the distribution of the activation scales of the model.
The model is loaded from the path specified in the `model_name` argument, and the activation scales are loaded from the path specified in the `output_file` argument.
The activation scales are then flattened and the distribution of the scales is calculated.
It will calculate the mean and standard deviation of the scales for each layer of the model before and after removing the top 0.1% of the scales.
"""

import argparse
import json
from utils.get_input_output_scales import get_static_decoder_layer_scales_distribution
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import numpy as np


def flatten_act_dict(act_dict):
    for layer, scales in act_dict.items():
        if isinstance(scales, list):
            try:
                all_acts = np.array(scales).reshape(-1)
            except ValueError:
                all_acts = [np.array(scale).reshape(-1) for scale in scales]
            all_acts = np.concatenate(all_acts)
            act_dict[layer] = all_acts
        else:
            act_dict[layer] = flatten_act_dict(scales)
            print(layer)
        gc.collect()

    return act_dict


def get_act_percentage(act_dict: dict, threshold: float):
    assert 0 <=threshold <= 1
    percentage = 1 - threshold
    act_percentage = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            all_acts_flattened = scales
            percentage_index = int(len(all_acts_flattened) * percentage) - 1
            nth_percentile_value = np.partition(all_acts_flattened, percentage_index)[percentage_index]
            act_percentage[layer] = float(nth_percentile_value)
        else:
            print(layer)
            act_percentage[layer] = get_act_percentage(scales, threshold)
    return act_percentage


def get_act_distribution_stat(act_dict):
    act_distribution = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            act_distribution[layer] = {'mean': float(np.mean(scales)), 'std': float(np.std(scales))}
        else:
            act_distribution[layer] = get_act_distribution_stat(scales)
            print(layer)
    return act_distribution


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument(
        "--dataset",
        type=argparse.FileType("r"),
        default="pile-val-backup/val.jsonl.zst",
    )
    parser.add_argument(
        "--output_file", type=str, default="act_scales_distribution.json"
    )
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, device_map="cuda")

    # You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst
    act_dict = get_static_decoder_layer_scales_distribution(
        model, tokenizer, args.dataset.name , num_samples=128
    )

    print("begin_flatten")
    act_dict = flatten_act_dict(act_dict)
    print("finish flatten")

    # origin model scale
    ori_scale = get_act_percentage(act_dict, 0)
    # scale after remove top 0.1% outliers
    top_0_1_scale = get_act_percentage(act_dict, 0.001)
    # get mean and std of all scales
    all_stat = get_act_distribution_stat(act_dict)

    res_dict = {"ori": ori_scale, "top_0_1": top_0_1_scale, "all_stat": all_stat}

    with open(args.output_file, "w") as f:
        json.dump(res_dict, f, indent=4, ensure_ascii=False)
