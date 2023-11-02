import os

# import ../../tools/convertor/converter.py
import sys

import torch

sys.path.append(os.path.realpath("../../tools/convertor"))
from converter import *

if __name__ == "__main__":
    writer = Writer("../../bin/quant_test.mllm")
    tensor_names = ["weight_f0", "weight_f1"]
    writer.write_tensor_index_padding(tensor_names)
    tensor0 = torch.zeros(
        (
            4096,
            3,
            512,
        )
    )
    tensor1 = torch.randn(
        (
            4096,
            3,
            512,
        )
    )
    writer.write_tensor(tensor0, tensor_names[0])
    writer.write_tensor(tensor0, tensor_names[1])
    writer.write_tensor_index()
