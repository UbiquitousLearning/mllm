from functools import partial
import gc
import json

import torch
import numpy as np

import argparse
import json

from model_interface import ModelFactory, ModelInterface


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
    assert 0 <= threshold <= 1
    percentage = 1 - threshold
    act_percentage = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            all_acts_flattened = scales
            percentage_index = int(len(all_acts_flattened) * percentage) - 1
            nth_percentile_value = np.partition(all_acts_flattened, percentage_index)[
                percentage_index
            ]
            act_percentage[layer] = float(nth_percentile_value)
        else:
            print(layer)
            act_percentage[layer] = get_act_percentage(scales, threshold)
    return act_percentage


@torch.no_grad()
def get_static_decoder_layer_scales_distribution(
    model_interface: ModelInterface,
    dataset_path,
    num_samples=32,
    no_bias=True,
):
    act_dict = {}
    
    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict:
            act_dict[name] = {}
        if "input" not in act_dict[name]:
            act_dict[name]["input"] = []
        act_dict[name]["input"].append(x.clone().detach().cpu().numpy())
        if isinstance(y, tuple):
            y = y[0]

        ty = y.clone().detach().cpu()
        # 去除 bias（只针对 nn.Linear）
        if no_bias and isinstance(m, torch.nn.Linear) and m.bias is not None:
            # print(name + str(".wobias"))
            # print(y.shape)
            
            bias = m.bias.clone().detach().view(1, -1)  # shape [1, out_features]
            ty = ty - bias.to(ty.device)

        if "output" not in act_dict[name]:
            act_dict[name]["output"] = []
        act_dict[name]["output"].append(ty.detach().cpu().numpy())

    hooks = []
    model_for_hook = model_interface.get_model_for_hook()
    for name, m in model_for_hook.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))
        
    print("Collecting activation scales...")

    from tqdm import tqdm
    
    dataset = model_interface.load_dataset(dataset_path, split="test")

    # 打乱数据集，设置随机种子以确保可重复性  
    shuffled_dataset = dataset.shuffle(seed=42)   

    processed_count = 0
    correct = 0

    with tqdm(total=num_samples) as pbar:
        
        pbar.set_description("Processing Dataset:")
        for data in shuffled_dataset:
            if model_interface.should_process_sample(data):
                # 进行推理
                inference_result = model_interface.infer(data)
                
                # 评估结果
                is_correct = model_interface.evaluate_sample(data, inference_result)
                if is_correct:
                    correct += 1
                else:
                    print(f"Sample failed: {data.get('file_name', 'unknown')}")
                
                processed_count += 1
                pbar.update(1)
            
            if processed_count >= num_samples:
                break
        
        if processed_count > 0:
            print(f"Accuracy: {correct / processed_count:.4f} ({correct}/{processed_count})")
        else:
            print("No samples were processed")
        

    for hook in hooks:
        hook.remove()

    return act_dict


def get_act_distribution_stat(act_dict):
    act_distribution = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            act_distribution[layer] = {
                "mean": float(np.mean(scales)),
                "std": float(np.std(scales)),
            }
        else:
            act_distribution[layer] = get_act_distribution_stat(scales)
    return act_distribution

from utils.config import ConfigDict, CONFIG_SCHEMA, validate_config

from pathlib import Path

def ensure_parent_dir(path: str | Path) -> None:
    path = Path(path)
    target_dir = path.parent if path.suffix else path
    target_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="mllm_qnn_convertor/config/qwen1.5-1.8b.json", help="Path to the config file")
    args = parser.parse_args()
    
    config = ConfigDict(json.load(open(args.config_file, "r")))
    config.check_schema(CONFIG_SCHEMA)
    validate_config(config)
    profile_config = config.profile_config

    profile_model_config = profile_config.model_config
    model_type = profile_model_config.model_type
    tokenizer_name = profile_model_config.tokenizer_name
    model_name = profile_model_config.model_name
    dataset_path = profile_config.dataset_path
    output_file = profile_config.output_path
    ensure_parent_dir(output_file)
    num_samples = profile_config.get("num_samples", 32)
    no_bias = profile_config.get("no_bias", True)
    

    print("=" * 60)
    print("Get Distribution Configuration:")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    print(f"Tokenizer Name: {tokenizer_name}")
    print(f"Model Name: {model_name}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Output File: {output_file}")
    print(f"Number of Samples: {num_samples}")
    print(f"No Bias: {no_bias}")
    print(f"Model Config: {dict(profile_model_config)}")
    print("=" * 60)
    print()
    
    if profile_model_config.save_rotation:
        ensure_parent_dir(profile_model_config.save_rotation)

    if profile_model_config.random_rotate and profile_model_config.R_path:
        raise ValueError("random_rotate and R_path cannot be true at the same time")

    model_interface = ModelFactory.create_model(
        model_type=model_type,
        tokenizer_name=tokenizer_name,
        model_name=model_name,
        args=profile_model_config
    )
    # FIXME: when num_samples is 1, this script will panic
    act_dict = get_static_decoder_layer_scales_distribution(model_interface, dataset_path, num_samples, no_bias)

    print("begin_flatten")
    act_dict = flatten_act_dict(act_dict)
    print("finish flatten")

    # origin model scale
    print("begin_calculate")
    print("get act 0")
    ori_scale = get_act_percentage(act_dict, 0)
    # scale after remove top 0.1% outliers
    print("get act 0.001")
    top_0_1_scale = get_act_percentage(act_dict, 0.001)
    # get mean and std of all scales
    print("get act distribution")
    all_stat = get_act_distribution_stat(act_dict)
    res_dict = {"ori": ori_scale, "top_0_1": top_0_1_scale, "all_stat": all_stat}
    with open(output_file, "w") as f:
        json.dump(res_dict, f, indent=4, ensure_ascii=False)

