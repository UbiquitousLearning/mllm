import argparse
from transformers import AutoModelForCausalLM
import torch
import json

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


@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8):
    # w: (out_features, in_features)
    w = w.to("cuda")
    scales = w.abs().max()
    q_max = 2 ** (n_bits - 1) - 1
    scales.clamp_(min=1e-5).div_(q_max)
    w.div_(scales).round_()

    if n_bits == 8:
        w = w.to("cpu").type(torch.int8)
    elif n_bits == 16:
        w = w.to("cpu").type(torch.int32)
    else:
        w = w.to("cpu").type(torch.int8)
    scale = scales.to("cpu").type(torch.float32)
    return w, scale


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
    parser.add_argument("--output_model", type=str, default="model-int8.pth")
    args = parser.parse_args()

    print("model: ", args.model_name)
    print("model type: ", args.model_type)
    print("scale file: ", args.scale_file.name)
    print("t01m clip threshold: ", args.t01m_clip_threshold)
    print("output model: ", args.output_model)

    model_name = args.model_name
    act_dict = args.scale_file.name
    t01m_clip_threshold = args.t01m_clip_threshold

    model = AutoModelForCausalLM.from_pretrained(model_name)
    act_dict = json.load(open(act_dict))

    act_scales, clip_top, return_dict = get_clip_and_scale(
        act_dict, t01m_clip_threshold
    )
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

    model_dict = q_model.state_dict()

    for i in act_scales:
        model_dict[i + ".input_scale"] = torch.tensor(act_scales[i]["input"])
        model_dict[i + ".output_scale"] = torch.tensor(act_scales[i]["output"])
        model_dict[i + ".clip_input"] = torch.tensor(clip_top[i]["input"])
        model_dict[i + ".clip_output"] = torch.tensor(clip_top[i]["output"])

    new_model = {}
    for name, param in model_dict.items():
        if name.replace(".weight", "") in act_scales:
            if "head" not in name:
                layer_name = name
                new_model[layer_name], scale = quantize_weight_per_tensor_absmax(
                    model_dict[layer_name], 8
                )
                new_model[layer_name + ".scale"] = scale

                # NOTE: the int8 weight used for QNN in mllm needs to be transposed
                new_model[name] = new_model[name].transpose(-2, -1)
                # print(f"Quantized {layer_name} with scale {scale}")
            else:
                new_model[name] = param
                # print(f"Copy {name}")
        elif name.replace(".bias", "") in act_scales:
            if "head" not in name:
                layer_name = name
                new_model[layer_name], scale = quantize_weight_per_tensor_absmax(
                    model_dict[layer_name], 8
                )
                new_model[layer_name + ".scale"] = scale
                # print(f"Quantized {layer_name} with scale {scale}")
            else:
                new_model[name] = param
                # print(f"Copy {name}")
        else:
            new_model[name] = param
            # print(f"Copy {name}")

    torch.save(new_model, args.output_model)
    print(f"Model saved to {args.output_model}")
