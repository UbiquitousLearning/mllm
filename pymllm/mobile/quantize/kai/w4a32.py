# Copyright (c) MLLM Team.
# Licensed under the MIT License.
from typing import Dict
import tvm_ffi
from ...ffi import (
    from_numpy,
    from_torch,
    Tensor,
    MLLM_FIND_NUMPY_AVAILABLE,
    MLLM_FIND_TORCH_AVAILABLE,
)
from ..quantize_pass import QuantizeBasePass, QuantizePlanPayload

if MLLM_FIND_TORCH_AVAILABLE:
    import torch
if MLLM_FIND_NUMPY_AVAILABLE:
    import numpy as np


class W4A32KAIQuantizePass(QuantizeBasePass):
    def __init__(self):
        super().__init__()

    def prepare(
        self, quantize_config, tensor_dict: Dict, **kwargs
    ) -> QuantizePlanPayload:
        replace: bool = quantize_config["replace"]
        rename: str = quantize_config["rename"] if not replace else None

        assert len(tensor_dict) in [1, 2]

        ret = QuantizePlanPayload()
        ret.inputs_num = len(tensor_dict)
        ret.inputs_dict = tensor_dict

        if replace:
            weight_key = None
            for key in tensor_dict.keys():
                if ".weight" in key:
                    weight_key = key
                    break
            ret.outputs_num = 1
            ret.outputs_dict = {weight_key: None}
        else:
            ret.outputs_num = 1
            ret.outputs_dict = {rename: None}
        return ret

    def match(self, quantize_config, tensor_dict: Dict, **kwargs) -> bool:
        if quantize_config["quant_method"] != "kai":
            return False
        if quantize_config["kai_matmul_triplet"] is None:
            return False
        if quantize_config["kai_matmul_triplet"] != "f32_qai8dxp_qsi4c32p":
            return False
        if quantize_config["kai_matmul_layout"] is None:
            return False
        if quantize_config["kai_matmul_layout"] != "mxk_nxk":
            return False
        return True

    def run(self, quantize_config, tensor_dict: Dict, **kwargs) -> Dict:
        normalized_tensor_dict = {}

        # Preprocess
        for k, v in tensor_dict.items():
            if isinstance(v, Tensor):
                normalized_tensor_dict.update({k: v})
            if MLLM_FIND_TORCH_AVAILABLE:
                if isinstance(v, torch.Tensor):
                    tmp_v: Tensor = from_torch(v.to(torch.float32))
                    tmp_v.set_name(k)
                    normalized_tensor_dict.update({k: tmp_v})
            if MLLM_FIND_NUMPY_AVAILABLE:
                if isinstance(v, np.ndarray):
                    tmp_v: Tensor = from_numpy(v.to(np.float32))
                    tmp_v.set_name(k)
                    normalized_tensor_dict.update({k: tmp_v})

        # Processing
        tile_cfg_name = quantize_config["kai_matmul_tile_cfg"]
        weight: Tensor = Tensor()
        bias: Tensor = Tensor()
        for key in normalized_tensor_dict.keys():
            if ".weight" in key:
                weight = normalized_tensor_dict[key]
            if ".bias" in key:
                bias = normalized_tensor_dict[key]
        weight: Tensor = tvm_ffi.get_global_func(
            "mllm.quantize_pack.KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk"
        )(tile_cfg_name, weight, bias)

        replace: bool = quantize_config["replace"]
        rename: str = quantize_config["rename"] if not replace else None

        if replace:
            return {weight.name: weight}
        else:
            weight.set_name(rename)
            return {rename: weight}
