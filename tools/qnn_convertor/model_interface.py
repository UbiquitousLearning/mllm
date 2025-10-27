from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union, Set
import torch
from datasets import Dataset


class ModelRegistry:
    """
    模型注册表，用于自动注册模型类型
    """
    _registry = {}
    
    @classmethod
    def register(cls, model_type: str):
        """
        装饰器，用于注册模型类型
        
        Args:
            model_type: 模型类型名称
        """
        def decorator(model_class):
            cls._registry[model_type] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_registry(cls) -> Dict[str, type]:
        """获取注册表"""
        return cls._registry.copy()
    
    @classmethod
    def get_model_class(cls, model_type: str) -> type:
        """根据模型类型获取模型类"""
        if model_type not in cls._registry:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(cls._registry.keys())}")
        return cls._registry[model_type]


class ModelInterface(ABC):
    """
    抽象基类，定义了模型接口的标准方法
    """
    
    def __init__(self, tokenizer_name: str, model_name: str, args: Any):
        """
        初始化模型接口
        
        Args:
            tokenizer_name: tokenizer名称或路径
            model_name: 模型名称或路径
            args: 配置参数
        """
        self.tokenizer_name = tokenizer_name
        self.model_name = model_name
        self.args = args
        self.model = None
        self._load_model()
        self.special_quantization_rules = {}
        self.skip_layers = {}
        if args.special_quantization_rules:
            self.special_quantization_rules = args.special_quantization_rules
        
        if args.skip_layers:
            self.skip_layers = args.skip_layers
    
    @abstractmethod
    def _load_model(self):
        """
        加载模型，需要在子类中实现
        """
        pass
    
    @abstractmethod
    def load_dataset(self, dataset_path: str, split: str = "test") -> Dataset:
        """
        加载数据集
        
        Args:
            dataset_path: 数据集路径
            split: 数据集分割（train/test/validation等）
            
        Returns:
            Dataset: 加载的数据集
        """
        pass
    
    @abstractmethod
    def infer(self, sample: Dict[str, Any]) -> Any:
        """
        对单个样本进行推理
        
        Args:
            sample: 数据集中的一个样本
            
        Returns:
            推理结果
        """
        pass
    
    @abstractmethod
    def evaluate_sample(self, sample: Dict[str, Any], inference_result: Any) -> bool:
        """
        评估单个样本的推理结果
        
        Args:
            sample: 数据集中的一个样本
            inference_result: 推理结果
            
        Returns:
            bool: 是否正确
        """
        pass
    
    @abstractmethod
    def should_process_sample(self, sample: Dict[str, Any]) -> bool:
        """
        判断是否应该处理这个样本（用于过滤）
        
        Args:
            sample: 数据集中的一个样本
            
        Returns:
            bool: 是否应该处理
        """
        pass
    
    def get_model_for_hook(self) -> torch.nn.Module:
        """
        获取用于注册hook的模型对象
        
        Returns:
            torch.nn.Module: 模型对象
        """
        return self.model
    
    def get_skip_layers(self) -> Dict[str, Set[str]]:
        """
        获取需要跳过的层
        
        Returns:
            Dict: {"skip_export": set(), "no_clip_input": set(), "no_clip_output": set()}
        """
        return {
            "skip_export": self.skip_layers.get("skip_export", set()) | self._get_skip_layers().get("skip_export", set()),
            "no_clip_input": self.skip_layers.get("no_clip_input", set()) | self._get_skip_layers().get("no_clip_input", set()),
            "no_clip_output": self.skip_layers.get("no_clip_output", set()) | self._get_skip_layers().get("no_clip_output", set()),
        }
    
    @abstractmethod
    def _get_skip_layers(self) -> Dict[str, Set[str]]:
        """
        获取需要跳过的层
        
        Returns:
            Dict: {"skip_export": set(), "no_clip_input": set(), "no_clip_output": set()}
        """
        pass
    
    def get_special_quantization_rules(self) -> Dict[str, Any]:
        """
        获取特殊量化规则
        
        Returns:
            Dict: 特殊量化规则
        """
        return {
            "skip_layers": self.special_quantization_rules.get("skip_layers", set()) | self._get_special_quantization_rules().get("skip_layers", set()),
            "head_layers": self.special_quantization_rules.get("head_layers", set()) | self._get_special_quantization_rules().get("head_layers", set()),
        }
    
    @abstractmethod
    def _get_special_quantization_rules(self) -> Dict[str, Any]:
        """
        获取特殊量化规则
        
        Returns:
            Dict: 特殊量化规则
        """
        pass
    
    def should_transpose_weight(self, layer_name: str) -> bool:
        """
        判断权重是否需要转置（QNN特定需求）
        
        Args:
            layer_name: 层名称
            
        Returns:
            bool: 是否需要转置
        """
        return True  # 默认需要转置


from transformers import AutoTokenizer, AutoModelForCausalLM
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import rotate

@ModelRegistry.register("qwen2")
class QwenModelInterface(ModelInterface):
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto")
        self.model.eval()
        if getattr(self.args, 'online_rotation', False):
            print("Online rotation enabled")
            if getattr(self.args, 'random_rotate', False):
                print("Using random rotation matrix")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                # model info
                num_layers = self.model.config.num_hidden_layers
                dim = self.model.config.hidden_size
                qo_heads = self.model.config.num_attention_heads
                head_dim = dim // qo_heads
                # get random hadamard rotation matrix
                R = rotate.get_orthogonal_matrix(dim, mode="hadamard", device=device)
                R_v = [rotate.get_orthogonal_matrix(head_dim, mode="hadamard" , device=device) for _ in range(num_layers)]
                
                if getattr(self.args, 'save_rotation', None):
                    R_bin = {
                        "R": R,
                        "R_v": R_v,
                    }
                    torch.save(R_bin, self.args.save_rotation)
                    print(f"Rotation matrix saved to {self.args.save_rotation}")
            else:
                print(f"Using pre-defined rotation matrix from {getattr(self.args, 'R_path', './R.bin')}")
                R_bin = torch.load(getattr(self.args, 'R_path', './R.bin'))
                R = R_bin["R"]
                R_v = R_bin["R_v"]
            
            print(f"Rotate model")
            rotate.rotate_model(self.model, R, R_v)
                
        
    def load_dataset(self, dataset_path: str, split: str = "test") -> Dataset:
        from datasets import load_dataset
        # ignore split since we only use one file
        dataset = load_dataset("json", data_files=f"{dataset_path}/val.jsonl.zst", split="train")
        return dataset
    
    def infer(self, sample: Dict[str, Any]) -> Any:
        with torch.no_grad():
            inputs = self.tokenizer(sample["text"][:6000], return_tensors="pt").to(self.model.device)
            # just simply forward
            self.model(**inputs)
        # don't return anything, just for profiling
        return None
    
    def evaluate_sample(self, sample: Dict[str, Any], inference_result: Any) -> bool:
        return True  # For profiling, we don't need to evaluate correctness

    def should_process_sample(self, sample: Dict[str, Any]) -> bool:
        return True
    
    def _get_skip_layers(self) -> Dict[str, Set[str]]:
        return {
            "skip_export": {"vision_tower"},  # 跳过视觉塔
            "no_clip_input": set(),
            "no_clip_output": set(),
        }
    
    def _get_special_quantization_rules(self) -> Dict[str, Any]:
        return {
            "skip_layers": {"lm_head", "merger"},  # 不量化的层
            "head_layers": {"head"},  # head层特殊处理
        }



@ModelRegistry.register("qwen2-vl")
class ShowUIModelInterface(ModelInterface):
    """
    ShowUI模型的具体实现
    """
    
    def _load_model(self):
        """加载ShowUI模型"""
        from utils.model import LLMNPUShowUIModel
        self.model = LLMNPUShowUIModel(self.tokenizer_name, self.model_name, args=self.args)
    
    def load_dataset(self, dataset_path: str, split: str = "test") -> Dataset:
        """加载ScreenSpot数据集"""
        from datasets import load_dataset
        return load_dataset(dataset_path, split=split)
    
    def infer(self, sample: Dict[str, Any]) -> Any:
        """对单个样本进行推理"""
        return self.model.infer(sample["image"], sample["instruction"], None)[0]
    
    def evaluate_sample(self, sample: Dict[str, Any], inference_result: Any) -> bool:
        """评估ScreenSpot样本的推理结果"""
        import ast
        try:
            point = ast.literal_eval(inference_result)
            bbox = sample["bbox"]
            x_min, y_min, x_max, y_max = bbox
            px, py = point
            is_inside = (x_min <= px <= x_max) and (y_min <= py <= y_max)
            return is_inside
        except:
            return False
    
    def should_process_sample(self, sample: Dict[str, Any]) -> bool:
        """判断是否应该处理ScreenSpot样本"""
        pc_or_mobile = sample["file_name"].split("_")[0]
        return sample["data_type"] in ["text"] and pc_or_mobile == "mobile"
    
    def get_model_for_hook(self) -> torch.nn.Module:
        """获取用于注册hook的模型对象"""
        return self.model.model
    
    def _get_skip_layers(self) -> Dict[str, Set[str]]:
        return {
            "skip_export": {"vision_tower"},
            "no_clip_input": set(),
            "no_clip_output": set(),
        }
    
    def _get_special_quantization_rules(self) -> Dict[str, Any]:
        return {
            "skip_layers": {"lm_head", "merger"},
            "head_layers": {"lm_head"},
        }


class ModelFactory:
    """
    模型工厂类，用于创建不同类型的模型接口
    """
    
    @classmethod
    def create_model(cls, model_type: str, tokenizer_name: str, model_name: str, args: Any) -> ModelInterface:
        """
        创建模型接口实例
        
        Args:
            model_type: 模型类型
            tokenizer_name: tokenizer名称或路径
            model_name: 模型名称或路径
            args: 配置参数
            
        Returns:
            ModelInterface: 模型接口实例
        """
        model_class = ModelRegistry.get_model_class(model_type)
        return model_class(tokenizer_name, model_name, args)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """
        获取所有可用的模型类型
        
        Returns:
            List[str]: 模型类型列表
        """
        return list(ModelRegistry.get_registry().keys())
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """
        手动注册模型类型（兼容旧代码）
        
        Args:
            model_type: 模型类型名称
            model_class: 模型类
        """
        ModelRegistry._registry[model_type] = model_class

if __name__ == "__main__":
    print("Available models:", ModelFactory.get_available_models())
