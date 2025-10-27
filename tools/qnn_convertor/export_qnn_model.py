import argparse
import torch
import json
from typing import Dict, Any

from model_interface import ModelFactory, ModelInterface
from utils.get_input_output_scales import get_clip_and_scale


class ModelExporter:
    """通用模型导出器"""
    
    def __init__(self, model_interface: ModelInterface, args):
        self.model_interface = model_interface
        self.args = args
        self.model = model_interface.get_model_for_hook()
    
    @torch.no_grad()
    def quantize_weight_per_tensor_absmax(self, w, n_bits=8):
        """权重量化"""
        w = w.to("cuda")
        scales = w.abs().max()
        q_max = 2 ** (n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_()

        if n_bits == 8:
            w = w.to("cpu").type(torch.int8)
        elif n_bits == 16 or n_bits == 32:
            w = w.to("cpu").type(torch.int32)
        else:
            w = w.to("cpu").type(torch.int8)
        scale = scales.to("cpu").type(torch.float32)
        return w, scale

    @torch.no_grad()
    def quantize_bias_per_tensor_absmax(self, w, n_bits=8):
        """bias量化"""
        w = w.to("cuda")
        scales = w.abs().max()
        q_max = 2 ** (n_bits - 1) - 1
        scales.clamp_(min=1e-5).div_(q_max)
        w.div_(scales).round_()

        w = w.to("cpu").type(torch.int32)
        scale = scales.to("cpu").type(torch.float32)
        return w, scale
    
    def get_activation_scales(self, act_dict_path: str):
        """获取激活量化参数"""
        act_dict = json.load(open(act_dict_path))
        
        skip_layers = self.model_interface.get_skip_layers()
        no_clip_input = skip_layers.get("no_clip_input", set())
        no_clip_output = skip_layers.get("no_clip_output", set())
        
        act_scales, clip_top, return_dict = get_clip_and_scale(
            act_dict, 
            self.args.t01m_clip_threshold, 
            self.args.clip_all, 
            no_clip_input=no_clip_input, 
            no_clip_output=no_clip_output
        )
        
        # 打印统计信息
        print(f"clip input num: {return_dict['clip_input_num']}")
        print(f"clip output num: {return_dict['clip_output_num']}")
        print(f"no clip input num: {return_dict['no_clip_input_num']}")
        for i in return_dict["no_clip_input_name"]:
            print(f"no clip input: {i}")
        print(f"no clip output num: {return_dict['no_clip_output_num']}")
        for i in return_dict["no_clip_output_name"]:
            print(f"no clip output: {i}")
            
        return act_scales, clip_top
    
    def should_skip_layer(self, name: str) -> bool:
        """判断是否应该跳过导出这个层"""
        skip_layers = self.model_interface.get_skip_layers()
        skip_patterns = skip_layers.get("skip_export", set())
        
        for pattern in skip_patterns:
            if pattern in name:
                return True
        return False
    
    def should_quantize_layer(self, name: str) -> bool:
        """判断是否应该量化这个层"""
        rules = self.model_interface.get_special_quantization_rules()
        skip_layers = rules.get("skip_layers", set())
        
        layer_name = name.replace(".weight", "").replace(".bias", "")
        for skip_pattern in skip_layers:
            if skip_pattern in layer_name:
                return False
        return True
    
    def is_head_layer(self, name: str) -> bool:
        """判断是否是head层"""
        rules = self.model_interface.get_special_quantization_rules()
        head_layers = rules.get("head_layers", set())
        
        for head_pattern in head_layers:
            if head_pattern in name:
                return True
        return False
    
    def export_model(self, act_dict_path: str, output_path: str):
        """导出量化模型"""
        
        # 获取激活量化参数
        act_scales, clip_top = self.get_activation_scales(act_dict_path)
        
        # 获取模型状态字典
        model_dict = self.model.state_dict()
        # move parameters to CPU
        for key in model_dict:
            model_dict[key] = model_dict[key].cpu()
        
        # 添加激活量化参数
        for layer_name in act_scales:
            model_dict[layer_name + ".input_scale"] = torch.tensor(act_scales[layer_name]["input"])
            model_dict[layer_name + ".output_scale"] = torch.tensor(act_scales[layer_name]["output"])
            print(f"{layer_name} input scale: {act_scales[layer_name]['input']}, output scale: {act_scales[layer_name]['output']}")
            model_dict[layer_name + ".clip_input"] = torch.tensor(clip_top[layer_name]["input"])
            model_dict[layer_name + ".clip_output"] = torch.tensor(clip_top[layer_name]["output"])
        
        # 量化和导出
        new_model = {}
        rules = self.model_interface.get_special_quantization_rules()
        
        for name, param in model_dict.items():
            print(name)
            
            # 跳过特定层
            if self.should_skip_layer(name):
                print(f"Skipping {name} as per skip rules")
                continue
            
            # 不量化的层直接复制
            if not self.should_quantize_layer(name):
                new_model[name] = param
                print(f"Skipping quantization for {name} as per special rules")
                continue
            
            # 权重量化
            if name.replace(".weight", "") in act_scales:
                if not self.is_head_layer(name):
                    layer_name = name
                    new_model[layer_name], scale = self.quantize_weight_per_tensor_absmax(param, 8)
                    new_model[layer_name + ".scale"] = scale
                    
                    # QNN需要转置权重
                    if self.model_interface.should_transpose_weight(layer_name):
                        new_model[name] = new_model[name].transpose(-2, -1)
                    
                    print(f"Quantized {layer_name} with scale {scale}")
                else:
                    new_model[name] = param
            
            # bias量化
            elif name.replace(".bias", "") in act_scales:
                if not self.is_head_layer(name):
                    layer_name = name
                    if not self.args.quant_bias:
                        new_model[name] = param
                        print(f"FP {layer_name}")
                    else:
                        new_model[layer_name], scale = self.quantize_bias_per_tensor_absmax(param, 8)
                        new_model[layer_name + ".scale"] = scale
                        print(f"Quantized {layer_name} with scale {scale}")
                else:
                    new_model[name] = param
            else:
                new_model[name] = param
        
        # 保存模型
        torch.save(new_model, output_path)
        print(f"Model saved to {output_path}")


from utils.config import ConfigDict, CONFIG_SCHEMA, validate_config

from pathlib import Path

def ensure_parent_dir(path: str | Path) -> None:
    path = Path(path)
    target_dir = path.parent if path.suffix else path
    target_dir.mkdir(parents=True, exist_ok=True)


MODEL_2_VIT_NAME = {
    "qwen2-vl": "visual"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    
    config = ConfigDict(json.load(open(args.config_file, "r")))
    config.check_schema(CONFIG_SCHEMA)
    validate_config(config)
    export_config = config.export_config
    
    model_config = export_config.model_config
    model_type = model_config.model_type
    tokenizer_name = model_config.tokenizer_name
    model_name = model_config.model_name
    scale_file = export_config.scale_file
    output_model = export_config.output_model
    
    assert export_config.quantize_vit is None or model_type in MODEL_2_VIT_NAME, \
        f"Model type {model_type} does not have quantization config for ViT"
    
    # when explicitly set quantize_vit to False
    # we skip the quantization of ViT layers
    # if not set or set to true, we don't skip any layer to quantize
    if export_config.quantize_vit == False:
        vit_name = MODEL_2_VIT_NAME.get(model_type, None)
        model_config["special_quantization_rules"] = {}
        model_config["special_quantization_rules"]["skip_layers"] = {vit_name}

    if model_config.random_rotate and model_config.R_path:
        raise ValueError("random_rotation and R_path cannot be true at the same time")
    
    ensure_parent_dir(output_model)
    
    print("model:", model_name)
    print("model type:", model_type)
    print("scale file:", scale_file)
    print("t01m clip threshold:", export_config.t01m_clip_threshold)
    print("output model:", output_model)
    print("Quantize bias:", export_config.quant_bias)
    print("quantize_vit:", export_config.quantize_vit)
    print(f"model config: {model_config}")
    
    # 创建模型接口
    model_interface = ModelFactory.create_model(
        model_type=model_type,
        tokenizer_name=tokenizer_name,
        model_name=model_name,
        args=model_config
    )
    
    # 创建导出器并导出模型
    exporter = ModelExporter(model_interface, export_config)
    exporter.export_model(scale_file, output_model)
