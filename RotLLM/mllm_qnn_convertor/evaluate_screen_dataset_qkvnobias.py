import torch
import argparse
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer, AutoProcessor
)
from copy import deepcopy

from utils.get_input_output_scales import get_clip_and_scale
from utils.quantization_simulation import (
    quantize_qwen2vl_qkvnobias_like,
)

from PIL import Image

import os
import json

import ast

from args import args

class LLMNPUQwen2Processor:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained(
            args.tokenizer_name,
            model_max_length=8192,
        )
        self.messages_template = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": None,
                },
                {"type": "text", "text": None},
            ],
        }


    def process(self, img: Image, text: str, json_path):

        messages = [deepcopy(self.messages_template)]
        messages[0]["content"][0]["image"] = ""
        messages[0]["content"][1]["text"] = text

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=img,
            padding=True,
            return_tensors="pt",
        )
        return inputs


class LLMNPUQWen2Model:
    def __init__(self, t01m_clip_threshold):

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=torch.float32, device_map="cuda", return_dict_in_generate=True,
        )
        act_dict = json.load(open(args.scale_file.name))

        act_scales, clip_top, return_dict = get_clip_and_scale(act_dict, t01m_clip_threshold)

        file_name = os.path.basename(args.model_name) + ".screenqa_profile.qkvnobias." + str(t01m_clip_threshold)  + ".clip.info"
        print(file_name)

        with open(file_name, "a") as f:  
            print(f"clip input num: {return_dict['clip_input_num']}", file=f)
            print(f"clip output num: {return_dict['clip_output_num']}", file=f)
            print(f"no clip input num: {return_dict['no_clip_input_num']}", file=f)
            for i in return_dict["no_clip_input_name"]:
                print(f"no clip input: {i}", file=f)
            print(f"no clip output num: {return_dict['no_clip_output_num']}", file=f)
            for i in return_dict["no_clip_output_name"]:
                print(f"no clip output: {i}", file=f)
        
        
        self.q_model = quantize_qwen2vl_qkvnobias_like(model, act_scales, layer_clip=clip_top)

        self.processor = LLMNPUQwen2Processor()

        self.t01m_clip_threshold = t01m_clip_threshold
        

    def infer(self, image: Image, text: str, json_path: str):
        inputs = self.processor.process(image, text, json_path)
        inputs = inputs.to("cuda")
        generated_ids = self.q_model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids["sequences"])
        ]

        output_text = self.processor.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        file_name = os.path.basename(args.model_name) + ".screenqa_profile.qkvnobias." + str(self.t01m_clip_threshold)  + ".output.text"
        # print(file_name)

        with open(file_name, "a") as f:  
            print(output_text,file=f)
        return output_text

class QWen2Model:
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=torch.float32, device_map="cuda", return_dict_in_generate=True,
        )

        self.processor = LLMNPUQwen2Processor()
        

    def infer(self, image: Image, text: str, json_path: str):
        inputs = self.processor.process(image, text, json_path)
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids["sequences"])
        ]

        output_text = self.processor.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # print(output_text)
        return output_text


from utils.model import LLMNPUShowUIModel


def perf_screenspot_showui(stage_2: bool = False):

    
    import ast
    from tqdm import tqdm
    from datasets import load_dataset

    dataset = load_dataset(args.dataset_path, split="test")  # noqa E501
    
    # thres = [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 10000, "clip_all"]
    # thres = [256, 10000, "clip_all"]
    thres = [128]
    if args.clip_all:
        thres = [1000000]

    if args.no_quantize:
        thres = [1]
        
    assert not (args.clip_all and args.no_quantize), "clip_all and no_quantize cannot be used together"

    for t01m_thre in thres:
        mobile_data_length = 0
        correct = 0
        
        if t01m_thre == "clip_all":
            t01m_thre = 1000000
            args.clip_all = True
        print("t01m_clip_threshold: ", t01m_thre)
        model = LLMNPUShowUIModel(args.tokenizer_name,
                                    args.model_name,
                                    args=args, t01m_clip_threshold=t01m_thre)

        with tqdm(total=len(dataset)) as pbar:
            pbar.set_description("ScreenSpotPipelineImpl Processing:")
            for data in dataset:
                pc_or_mobile = data["file_name"].split("_")[0]
                if data["data_type"] in ["icon"] and pc_or_mobile == "mobile":
                    mobile_data_length += 1
                    point = []
                    out_text = ""
                    try:
                        out_text = model.infer(data["image"], data["instruction"], None)[0]
                        point = ast.literal_eval(out_text)
                    except:
                        print(data["file_name"], "ast parse failed", out_text)
                        pbar.update(1)
                        continue
                    bbox = data["bbox"]
                    x_min, y_min, x_max, y_max = bbox
                    px, py = point
                    is_inside = (x_min <= px <= x_max) and (y_min <= py <= y_max)
                    if is_inside:
                        correct += 1
                    else:
                        print(data["file_name"], "position failed", bbox, point)
                pbar.update(1)

        file_name = os.path.basename(args.model_name) + "." + os.path.basename(args.scale_file.name) + ".screenspot.icon.qkvnobias.mllmsetting." + str(t01m_thre)  + ".result.info"
        with open(file_name, "a") as f:  
            print("acc: ", correct / mobile_data_length, file=f)


if __name__ == "__main__":
    perf_screenspot_showui()
    