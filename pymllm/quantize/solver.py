# Copyright (c) MLLM Team.
# Licensed under the MIT License.
import re
from .quantize_pass import QuantizeBasePass, QuantizePlanPayload
from ..convertor import ModelFileV2
from typing import Dict, List, Any
from ..ffi import MLLM_FIND_NUMPY_AVAILABLE, MLLM_FIND_TORCH_AVAILABLE

if MLLM_FIND_TORCH_AVAILABLE:
    import torch
if MLLM_FIND_NUMPY_AVAILABLE:
    import numpy as np


class QuantizeSolver:
    def __init__(self):
        self.passes: List[QuantizeBasePass] = []

    def register_pass(self, pass_: QuantizeBasePass):
        self.passes.append(pass_)

    def _stream_quantize_write_v2(self, tensor_dict: Dict, writer: ModelFileV2) -> bool:
        pass

    def stream_quantize_params_size(
        self, quant_cfg, tensor_dict: Dict, **kwargs
    ) -> int:
        param_groups: Dict[str, List[Any, Dict]] = {}
        for k, v in quant_cfg.items():
            sub_group: Dict[str, QuantizePlanPayload] = {}
            hints = v["hints"]
            pattern = re.compile(k)
            for pk, pv in tensor_dict.items():
                if pattern.fullmatch(pk) is not None:
                    # pk is model.linear_0.weight or model.linear_0.bias
                    # layer_name is model.linear_0
                    layer_name, _ = pk.rsplit(".", 1)
                    if layer_name not in sub_group:
                        sub_group[layer_name] = QuantizePlanPayload()
                    sub_group[layer_name].inputs_num += 1
                    sub_group[layer_name].inputs_dict.update({pk: pv})
            param_groups.update({k: [hints, sub_group]})

        # Prepare inputs and outputs, calculate the params
        for group_regex in param_groups.keys():
            sub_group = param_groups[group_regex]
            for payload_name, payload in sub_group[1].items():
                for pass_ in self.passes:
                    if pass_.match(param_groups[group_regex][0], payload.inputs_dict):
                        prepared_payload = pass_.prepare(
                            param_groups[group_regex][0],
                            payload.inputs_dict,
                        )
                        param_groups[group_regex][1][payload_name] = prepared_payload
                        break

        # Update params nums
        aux = set(tensor_dict.keys())

        for group_regex in param_groups.keys():
            sub_group = param_groups[group_regex]
            for payload_name, payload in sub_group[1].items():
                if param_groups[group_regex][0]["replace"]:
                    for k in payload.inputs_dict.keys():
                        aux.remove(k)
                    for k in payload.outputs_dict.keys():
                        aux.add(k)
                else:
                    for k in payload.outputs_dict.keys():
                        aux.add(k)

        return len(aux)

    def stream_quantize(
        self, quant_cfg, tensor_dict: Dict, writer: ModelFileV2, **kwargs
    ) -> bool:
        if not isinstance(writer, ModelFileV2):
            raise NotImplementedError(
                "stream_quantize only support type: ModelFileV2 currently."
            )

        # Planning
        param_groups: Dict[str, List[Any, Dict]] = {}
        for k, v in quant_cfg.items():
            sub_group: Dict[str, QuantizePlanPayload] = {}
            hints = v["hints"]
            pattern = re.compile(k)
            for pk, pv in tensor_dict.items():
                if pattern.fullmatch(pk) is not None:
                    # pk is model.linear_0.weight or model.linear_0.bias
                    # layer_name is model.linear_0
                    layer_name, _ = pk.rsplit(".", 1)
                    if layer_name not in sub_group:
                        sub_group[layer_name] = QuantizePlanPayload()
                    sub_group[layer_name].inputs_num += 1
                    sub_group[layer_name].inputs_dict.update({pk: pv})
            param_groups.update({k: [hints, sub_group]})

        # Prepare inputs and outputs, calculate the params
        for group_regex in param_groups.keys():
            sub_group = param_groups[group_regex]
            for payload_name, payload in sub_group[1].items():
                for pass_ in self.passes:
                    if pass_.match(param_groups[group_regex][0], payload.inputs_dict):
                        prepared_payload = pass_.prepare(
                            param_groups[group_regex][0],
                            payload.inputs_dict,
                        )
                        param_groups[group_regex][1][payload_name] = prepared_payload
                        break

        # Show Planned Info
        verbose = kwargs.get("verbose", False)
        if verbose:
            print("Planned Quantized Info:")
            for group_regex in param_groups.keys():
                print(f"{group_regex}:")
                sub_group = param_groups[group_regex]
                for payload_name, payload in sub_group[1].items():
                    print(" " * 4 + payload_name + ":")
                    print(" " * 8 + f"inputs num: {payload.inputs_num}")
                    print(" " * 8 + f"outputs num: {payload.outputs_num}")
                    print(" " * 8 + "params before quantization:")
                    for k in payload.inputs_dict.keys():
                        print(" " * 12 + k)
                    print(" " * 8 + "params after quantization:")
                    for k in payload.outputs_dict.keys():
                        print(" " * 12 + k)

        # Processing
        left_name = set(tensor_dict.keys())
        for group_regex in param_groups.keys():
            sub_group = param_groups[group_regex]
            for payload_name, payload in sub_group[1].items():
                for pass_ in self.passes:
                    if pass_.match(param_groups[group_regex][0], payload.inputs_dict):
                        prepared_payload = pass_.run(
                            param_groups[group_regex][0],
                            payload.inputs_dict,
                        )
                        if param_groups[group_regex][0]["replace"]:
                            for pk in payload.inputs_dict.keys():
                                left_name.remove(pk)
                                tensor_dict[pk] = None
                            for pk, pv in prepared_payload.items():
                                if verbose:
                                    print(pk)
                                writer.streaming_write(pk, pv)
                        else:
                            for pk, pv in prepared_payload.items():
                                writer.streaming_write(pk, pv)
                payload.inputs_dict = None
                payload.outputs_dict = None

        cast_left_2_fp32 = kwargs.get("cast_left_2_fp32", False)

        for k in left_name:
            if verbose:
                print(k)
            ttt = tensor_dict[k]
            if cast_left_2_fp32:
                if MLLM_FIND_TORCH_AVAILABLE and isinstance(ttt, torch.Tensor):
                    ttt = ttt.to(torch.float32)
                elif MLLM_FIND_NUMPY_AVAILABLE and isinstance(ttt, np.float32):
                    ttt = ttt.to(np.float32)
            writer.streaming_write(k, ttt)

        writer.finalize()
