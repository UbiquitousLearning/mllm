from collections import defaultdict
import torch
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
import numpy as np
from utils.quantization_simulation import W8A8LinearStatic


@torch.no_grad()
def get_static_decoder_layer_scales_distribution(
    model,
    tokenizer,
    dataset_path,
    num_samples=32,
    seq_len=512,
):
    """
    Get the distribution of the input and output scales of the model's layers.
    Including the original scale, the scale after removing the top 0.1% outliers,
    and the mean and standard deviation of the scales.
    """
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = []
        act_dict[name]["input"].append(x.detach().cpu().numpy())
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = []
        act_dict[name]["output"].append(y.detach().cpu().numpy())

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()

    return act_dict


def get_clip_and_scale(act_dict: dict, t01m_thre=5) -> tuple:
    """
    Get the clipped(W8A8) and no clipped(shadow linear to restore origin scale) input and output scales of the model's layers.
    """
    top_0_1 = act_dict["top_0_1"]
    ori_scale = act_dict["ori"]
    stat = act_dict["all_stat"]
    act_scale = {}
    clip_top = {}
    clip_input_num = 0
    no_clip_input_num = 0
    clip_output_num = 0
    no_clip_output_num = 0
    no_clip_input_name = []
    no_clip_output_name = []

    for i in stat:
        top_0_1_input = top_0_1[i]["input"]
        top_0_1_output = top_0_1[i]["output"]
        act_scale[i] = {}
        clip_top[i] = {}
        # layer input
        if top_0_1_input * t01m_thre > ori_scale[i]["input"]:
            clip_input_num += 1
            clip_top[i]["input"] = True
            act_scale[i]["input"] = ori_scale[i]["input"]
        else:
            no_clip_input_num += 1
            clip_top[i]["input"] = False
            act_scale[i]["input"] = top_0_1[i]["input"]
            no_clip_input_name.append(i)
        # layer output
        if top_0_1_output * t01m_thre > ori_scale[i]["output"]:
            clip_output_num += 1
            clip_top[i]["output"] = True
            act_scale[i]["output"] = ori_scale[i]["output"]
        else:
            no_clip_output_num += 1
            clip_top[i]["output"] = False
            act_scale[i]["output"] = top_0_1[i]["output"]
            no_clip_output_name.append(i)

    return_dict = {
        "t01m_thre": t01m_thre,
        "clip_input_num": clip_input_num,
        "no_clip_input_num": no_clip_input_num,
        "clip_output_num": clip_output_num,
        "no_clip_output_num": no_clip_output_num,
        "no_clip_input_name": no_clip_input_name,
        "no_clip_output_name": no_clip_output_name,
    }
    return act_scale, clip_top, return_dict
