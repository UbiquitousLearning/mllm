import argparse
import json
import struct
from functools import reduce
from io import BufferedWriter
import os
import torch
MAGIC_NUMBER = 20012
file_map = {}
class Tensor:
    name: str
    offset: int
    size: int
    dtype: int
    def __init__(self, name: str, dtype: int):
        self.name = name
        self.dtype = dtype
# One Tensor Index Item Contains: Name_Len(Int)+Name(str)+Weights_Len(UInt64)+Offset(UInt64)+DataType(Int)
def calc_tensors_index_table_size(name: str):
    return 4 + len(name) + 8 + 8 + 4
class Writer:
    writer: BufferedWriter
    tensors_map: [str, Tensor]
    tensors_name: [str]

    def __init__(self, path: str):
        self.tensors_map = {}
        self.tensors_name = []
        self.writer = open(path, "wb+")
        self.writer.seek(0)
        self.write_int(MAGIC_NUMBER)
    def __torch_dtype_to_int(self, dtype: torch.dtype) -> int:
        if dtype == torch.float32 or dtype == torch.bfloat16:
            return 0
        elif dtype == torch.float16:
            return 1
        elif dtype == torch.int8:
            return 16
        elif dtype == torch.int8:
            return 17
        elif dtype == torch.int32:
            return 18
        else:
            raise Exception(f"Unknown dtype: {dtype}")
    def write_int(self, val: int):
        self.writer.write(struct.pack("<i", val))
    def write_float(self, val: float):
        self.writer.write(struct.pack("<f", val))
    def write_u64(self, val: int):
        self.writer.write(struct.pack("<Q", val))
    def write_str(self, val: str):
        self.writer.write(struct.pack("<i", len(val)))
        self.writer.write(val.encode("utf-8"))

    def write_tensor(self, tensor: torch.Tensor, name: str) -> [int, int]:
        tensor_idx = Tensor(name=name, dtype=self.__torch_dtype_to_int(tensor.dtype))
        self.tensors_map[name] = tensor_idx
        offset = self.writer.tell()
        if tensor.dtype == torch.bfloat16:  # to float 16
            tensor_numpy = tensor.detach().to(torch.float16).numpy()
        else:
            tensor_numpy = tensor.numpy()
        tensor_numpy.tofile(self.writer)
        size = self.writer.tell() - offset
        tensor_idx.size = size
        tensor_idx.offset = offset
        return offset, size
    def write_tensor_index(
            self,
    ):
        self.writer.seek(4 + 8)
        for tensor_name in self.tensors_name:
            tensor = self.tensors_map[tensor_name]
            # self.write_int(len(tensor.name))
            tensor.name = tensor.name.replace("_weight", ".weight")
            tensor.name = tensor.name.replace("_bias", ".bias")
            # todo: nort used in GTEST
            # tensor.name = key_map(tensor.name, args.type)
            self.write_str(tensor.name)
            self.write_u64(tensor.size)
            self.write_u64(tensor.offset)
            self.write_int(tensor.dtype)
            print(f"Write tensor {tensor.name} to {tensor.offset} with size {tensor.size}")

    def write_tensor_index_padding(self, tensors_name: [str]):
        if len(tensors_name) > 0:
            self.tensors_name = tensors_name
            padding_size = reduce(
                lambda x, y: x + y, map(calc_tensors_index_table_size, tensors_name)
            )
            self.writer.seek(4)
            self.write_u64(padding_size)
            print(f"Padding size: {padding_size}")
            self.writer.write(b"\x00" * padding_size)
            self.writer.flush()
            return
        else:
            raise Exception("No tensors to write")
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='convert pytorch predictor to mllm model')
    argparser.add_argument('--input', type=str, default='mlp_predictor')
    argparser.add_argument('--output_model', type=str, default='model.mllm')
    args = argparser.parse_args()

    index_path = os.path.join(args.input, 'index.json')
    with open(index_path, 'r') as f:
        index = json.load(f)

    all_tensors = {}
    for item in index:
        prefix = item['prefix']
        for weight, file in item.items():
            if weight == 'prefix':
                continue
            name = '.'.join((prefix, weight))
            file_path = os.path.join(args.input, file)
            state_dict = torch.load(file_path)
            all_tensors[name] = state_dict[weight]

    writer = Writer(args.output_model)
    model_keys = list(all_tensors.keys())
    writer.write_tensor_index_padding(model_keys)

    for key in model_keys:
        tensor = all_tensors[key]
        offset, size = writer.write_tensor(tensor, key)
        print(f"Get tensor {key} to {offset} with size {size}")

    writer.write_tensor_index()

