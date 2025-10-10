# Copyright (c) MLLM Team.
# Licensed under the MIT License.
from typing import Dict
from .solver import QuantizeSolver

# Include all passes
from .kai.w4a32 import W4A32KAIQuantizePass
from .cast2fp32_pass import Cast2Fp32QuantizePass


def build_w4a32_kai_pipeline() -> QuantizeSolver:
    ret = QuantizeSolver()
    ret.register_pass(W4A32KAIQuantizePass())
    # ret.register_pass(Cast2Fp32QuantizePass())
    return ret


BUILTIN_QUANTIZE_PIPELINE: Dict = {"w4a32_kai_pipeline": build_w4a32_kai_pipeline}
BUILTIN_QUANTIZE_PASS: Dict = {
    "w4a32_kai": W4A32KAIQuantizePass,
    "cast2fp32": Cast2Fp32QuantizePass,
}
