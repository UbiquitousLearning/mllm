from collections import defaultdict
import torch
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
import numpy as np
from fake_quant_dynamic_clip import W8A8LinearStatic

@torch.no_grad()
def get_static_decoder_layer_scales(
    model,
    tokenizer,
    dataset_path,
    num_samples=512,
    seq_len=512,
):
    model.eval()
    device = next(model.parameters()).device

    act_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict or "input" not in act_dict[name]:
            act_dict[name]["input"] = x.detach().abs().max().item()
        else:
            act_dict[name]["input"] = max(
                act_dict[name]["input"], x.detach().abs().max().item()
            )
        if isinstance(y, tuple):
            y = y[0]
        if name not in act_dict or "output" not in act_dict[name]:
            act_dict[name]["output"] = y.detach().abs().max().item()
        else:
            act_dict[name]["output"] = max(
                act_dict[name]["output"], y.detach().abs().max().item()
            )

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
        mean_scale = np.mean([v["input"] for v in act_dict.values()])
        pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()
        
    return act_dict



@torch.no_grad()
def get_static_decoder_layer_scales_distribution(
    model,
    tokenizer,
    dataset_path,
    num_samples=32,
    seq_len=512,
):
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


@torch.no_grad()
def get_outlier_distribution(
    model,
    tokenizer,
    act_dict,
    num_samples=4,
    seq_len=512,
):
    model.eval()
    device = next(model.parameters()).device

    outlier_dict = defaultdict(dict)

    def stat_io_hook(m, x, y, name):
        print('call_hook')
        if isinstance(x, tuple):
            x = x[0]
        if name not in outlier_dict or "input" not in outlier_dict[name]:
            outlier_dict[name]["input"] = []
        threshold = act_dict[name]["input"]
        print(name)
        print(x.shape)
        x = x.detach().cpu().numpy()
        print(threshold, x.max())
        outliers = [(i, val) for i, val in np.ndenumerate(x) if abs(val) > threshold]
        outlier_dict[name]["input"].append((outliers, x.shape))
        print(len(outliers), len(x.flatten()))
        if isinstance(y, tuple):
            y = y[0]
        print(y.shape)
        if name not in outlier_dict or "output" not in outlier_dict[name]:
            outlier_dict[name]["output"] = []
        threshold = act_dict[name]["output"]
        print(threshold, y.max())
        y = y.detach().cpu().numpy()
        outliers = [(i, val) for i, val in np.ndenumerate(y) if abs(val) > threshold]
        outlier_dict[name]["output"].append((outliers, y.shape))
        print(len(outliers), len(y.flatten()))

    hooks = []
    for name, m in model.named_modules():
        print(name, type(m))
        if isinstance(m, W8A8LinearStatic):
            hooks.append(m.register_forward_hook(partial(stat_io_hook, name=name)))

    print("Collecting activation scales...")
    pbar = tqdm(range(num_samples))
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)
    for i in pbar:
        batch = dataset[:, (i * 512) : ((i + 1) * 512)].to(model.device)
        model(batch)
        #mean_scale = np.mean([v["input"] for v in act_dict.values()])
        #pbar.set_description(f"Mean input scale: {mean_scale:.2f}")
    for hook in hooks:
        hook.remove()
        
    return outlier_dict