# Need a HUGE amount of memory
from get_input_output_scales import get_static_decoder_layer_scales_distribution
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import numpy as np


model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-1.8B-Chat', device_map='cuda:0')
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device_map='cuda:0')

#You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst
act_dict = get_static_decoder_layer_scales_distribution(model, tokenizer, 'dataset/val.jsonl.zst', num_samples=40)

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

print('begin_flatten')
act_dict = flatten_act_dict(act_dict)
print('finish flatten')

def get_act_percentage(act_dict, top_percentage):
    assert 0 <=top_percentage <= 1
    percentage = 1 - top_percentage
    act_percentage = {}
    for layer, scales in act_dict.items():
        if not isinstance(scales, dict):
            all_acts_flattened = scales
            percentage_index = int(len(all_acts_flattened) * percentage) - 1
            nth_percentile_value = np.partition(all_acts_flattened, percentage_index)[percentage_index]
            act_percentage[layer] = float(nth_percentile_value)

        else:
            print(layer)
            act_percentage[layer] = get_act_percentage(scales, top_percentage)
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

ori_scale = get_act_percentage(act_dict, 0)
print('-------------------')
top_0_1_scale = get_act_percentage(act_dict, 0.001)
print('-------------------')
top_0_25_scale = get_act_percentage(act_dict, 0.0025)
print('-------------------')
top_0_5_scale = get_act_percentage(act_dict, 0.005)
print('-------------------')
top_1_scale = get_act_percentage(act_dict, 0.01)
print('-------------------')
top_3_scale = get_act_percentage(act_dict, 0.03)
print('-------------------')
top_5_scale = get_act_percentage(act_dict, 0.05)
print('-------------------')
top_10_scale = get_act_percentage(act_dict, 0.1)
print('-------------------')
top_30_scale = get_act_percentage(act_dict, 0.3)
print('-------------------')
all_stat = get_act_distribution_stat(act_dict)

res_dict = {
    'ori': ori_scale,
    'top_0_1': top_0_1_scale,
    'top_0_25': top_0_25_scale,
    'top_0_5': top_0_5_scale,
    'top_1': top_1_scale,
    'top_3': top_3_scale,
    'top_10': top_10_scale,
    'top_30': top_30_scale,
    'all_stat': all_stat
}
import json
with open('qwen1.5-1.8b_act_scales_distribution.json', 'w') as f:
    json.dump(res_dict, f, indent=4, ensure_ascii=False)