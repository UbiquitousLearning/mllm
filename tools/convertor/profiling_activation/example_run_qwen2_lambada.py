from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import multiprocessing
import json
import numpy as np
from tqdm import tqdm

ori_models = {
    "qwen1.5-1.8b": ["Qwen/Qwen1.5-1.8B-Chat", "./qwen1.5-1.8b_act_scales_distribution.json"]
}

from fake_quant_dynamic_clip import quantize_qwen2_like, quantize_llama_like, quantize_gemma_like, quantize_opt, quantize_phi_like


def get_clip_and_scale(act_dict, t01m_thre = 5):
    top_0_1 = act_dict['top_0_1']
    ori_scale = act_dict['ori']
    stat = act_dict['all_stat']
    act_scale = {}
    clip_top = {}
    clip_input_num = 0
    no_clip_input_num = 0
    clip_output_num = 0
    no_clip_output_num = 0
    for i in stat:
        top_0_1_input = top_0_1[i]['input']
        top_0_1_output = top_0_1[i]['output']
        act_scale[i] = {}
        clip_top[i] = {}

        if top_0_1_input * t01m_thre > ori_scale[i]['input']:
            clip_input_num += 1
            print(i, top_0_1_input, ori_scale[i]['output'], 'clip')
            clip_top[i]['input'] = True
            act_scale[i]['input'] = ori_scale[i]['input']
        else:
            no_clip_input_num += 1
            clip_top[i]['input'] = False
            act_scale[i]['input'] = top_0_1[i]['input']
        if top_0_1_output * t01m_thre > ori_scale[i]['output']:
            clip_output_num += 1
            print(i, top_0_1_output, ori_scale[i]['output'], 'clip')
            clip_top[i]['output'] = True
            act_scale[i]['output'] = ori_scale[i]['output']
        else:
            no_clip_output_num += 1
            clip_top[i]['output'] = False
            act_scale[i]['output'] = top_0_1[i]['output']
    print('clip_input_num:', clip_input_num, 'no_clip_input_num:', no_clip_input_num)
    print('clip_output_num:', clip_output_num, 'no_clip_output_num:', no_clip_output_num)
    return_dict = {
        't01m_thre': t01m_thre,
        'clip_input_num': clip_input_num,
        'no_clip_input_num': no_clip_input_num,
        'clip_output_num': clip_output_num,
        'no_clip_output_num': no_clip_output_num
    }
    return act_scale, clip_top, return_dict

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
    
    
from datasets import load_dataset
    
    
def evaluate_model(model_name, act_dict, result_queue, t01m_thre):
    dataset = load_dataset('lambada', split="validation[:1000]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='cuda:1')
    evaluator = Evaluator(dataset, tokenizer, "cuda:1")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='cuda:1')

    act_scales, clip_top, return_dict = get_clip_and_scale(act_dict, t01m_thre)

    if model_name == "Qwen/Qwen1.5-1.8B-Chat":
        q_model = quantize_qwen2_like(model, act_scales,layer_clip=clip_top)
        
    res = evaluator.evaluate(q_model)
    print(t01m_thre, res)

    return_dict['res'] = float(res)
    result_queue.put((t01m_thre, return_dict))


def get_all_actscale_result_parallel(model_name, act_dict):
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    for t01m_thre in [1,2,3,4,5,8,10,12,16,20,24,30,32,10000000]:
        p = multiprocessing.Process(target=evaluate_model, args=(model_name, act_dict, result_queue, t01m_thre))
        p.start()
        p.join()
    

    results = {}
    while not result_queue.empty():
        key, value = result_queue.get()
        results[key] = value

    return results

res_data = {}
for model_name in ori_models:
    #try:
    model_info = ori_models[model_name]
    model_name = model_info[0]
    act_dict = json.load(open(model_info[1]))
    results = get_all_actscale_result_parallel(model_name, act_dict)
    res_data[model_name] = results

print(res_data)
with open('all_model_res_lambada_new.json', 'w') as f:
    json.dump(res_data, f, indent=4, ensure_ascii=False)
