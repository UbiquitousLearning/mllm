# Copyright (c) MLLM Team.
# Licensed under the MIT License.
from typing import Dict
from .quantize_pass import QuantizeBasePass, QuantizePlanPayload
from ..ffi import (
    Tensor,
    MLLM_FIND_NUMPY_AVAILABLE,
    MLLM_FIND_TORCH_AVAILABLE,
)

if MLLM_FIND_TORCH_AVAILABLE:
    import torch
if MLLM_FIND_NUMPY_AVAILABLE:
    import numpy as np


class Cast2Fp32QuantizePass(QuantizeBasePass):
    def __init__(self):
        super().__init__()

    def prepare(
        self, quantize_config, tensor_dict: Dict, **kwargs
    ) -> QuantizePlanPayload:
        assert len(tensor_dict) == 1
        return QuantizePlanPayload(
            1,
            1,
            {
                tensor_dict.keys()[0]: tensor_dict[tensor_dict.keys()[0]],
            },
            {
                tensor_dict.keys()[0],
                None,
            },
        )

    def match(self, quantize_config, tensor_dict: Dict, **kwargs) -> bool:
        if quantize_config["quant_method"] != "cast_2_fp32":
            return False
        ret = False
        for k, v in tensor_dict.items():
            if isinstance(v, torch.Tensor):
                if v.dtype is not torch.float32:
                    ret = True
            if isinstance(v, np.ndarray):
                if v.dtype is not np.float32:
                    ret = True
        return ret

    def run(self, quantize_config, tensor_dict: Dict, **kwargs) -> Dict:
        name = tensor_dict.keys()[0]
        weight = tensor_dict[name]
        if isinstance(weight, torch.Tensor):
            weight = weight.to(torch.float32)
        if isinstance(weight, np.ndarray):
            weight = weight.to(np.float32)
        return {name: weight}
