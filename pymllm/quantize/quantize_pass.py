# Copyright (c) MLLM Team.
# Licensed under the MIT License.
from typing import Dict
from abc import ABC, abstractmethod


class QuantizePlanPayload:
    def __init__(self):
        self.inputs_num: int = 0
        self.outputs_num: int = 0
        self.inputs_dict: Dict = {}
        self.outputs_dict: Dict = {}


class QuantizeBasePass(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def prepare(
        self, quantize_config, tensor_dict: Dict, **kwargs
    ) -> QuantizePlanPayload:
        pass

    @abstractmethod
    def match(self, quantize_config, tensor_dict: Dict, **kwargs) -> bool:
        pass

    @abstractmethod
    def run(self, quantize_config, tensor_dict: Dict, **kwargs) -> Dict:
        pass
