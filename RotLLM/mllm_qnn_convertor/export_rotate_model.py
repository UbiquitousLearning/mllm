import argparse
import torch
import json

from model_interface import ModelFactory


class RotateModelExporter:
    """旋转模型导出器 - 简化版，只导出state_dict"""
    
    def __init__(self, model_interface):
        self.model_interface = model_interface
        self.model = model_interface.get_model_for_hook()
    
    def export_model(self, output_path: str):
        """导出模型state_dict"""
        print("Getting model state dict...")
        
        # 获取模型状态字典
        model_dict = self.model.state_dict()
        
        # 移动到CPU（如果在GPU上）
        print("Moving parameters to CPU...")
        for key in model_dict:
            model_dict[key] = model_dict[key].cpu()
        
        # 保存模型
        print(f"Saving model to {output_path}...")
        torch.save(model_dict, output_path)
        print(f"Model successfully saved to {output_path}")


from utils import ConfigDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    
    # 从配置文件加载参数
    config = json.load(open(args.config_file, "r"))
    
    model_type = config["model_type"]
    tokenizer_name = config["tokenizer_name"]
    model_name = config["model_name"]
    output_model = config.get("output_model", "rotated_model.bin")
    
    # 将配置转换为可点号访问的对象
    model_config = ConfigDict(config.get("model_config", {}))
    
    print("=" * 50)
    print("Rotate Model Export Configuration")
    print("=" * 50)
    print(f"Model type: {model_type}")
    print(f"Model name: {model_name}")
    print(f"Tokenizer name: {tokenizer_name}")
    print(f"Output model: {output_model}")
    print(f"Model config: {model_config}")
    print("=" * 50)
    
    # 创建模型接口（会自动应用旋转等操作）
    print("Creating model interface...")
    model_interface = ModelFactory.create_model(
        model_type=model_type,
        tokenizer_name=tokenizer_name,
        model_name=model_name,
        args=model_config
    )
    
    # 创建导出器并导出模型
    print("Creating exporter...")
    exporter = RotateModelExporter(model_interface)
    exporter.export_model(output_model)
    
    print("Export completed successfully!")
    