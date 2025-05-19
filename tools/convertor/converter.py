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
        elif dtype == torch.int8 or dtype == torch.bool:
            return 16
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
            tensor_numpy = tensor.detach().to(torch.float32).numpy()
        elif tensor.dtype == torch.bool or tensor.dtype == torch.int8:  # exported model for QNN int8
            tensor_numpy = tensor.detach().to(torch.int8).numpy()
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


def get_tensor(model: dict, key: str, index_: dict):
    if index_ is not None and isinstance(index_, dict) and "weight_map" in index_.keys():
        if key in index_["weight_map"].keys():
            model_ = file_map[index_["weight_map"][key]]
            if args.type == "torch":
                return model_[key]
            if args.type == "safetensor":
                return model_.get_tensor(key)
        else:
            raise Exception(f"Tensor {key} not found in index")
    if key in model.keys():
        if args.type == "torch":
            return model[key]
        if args.type == "safetensor":
            return model.get_tensor(key)
    else:
        raise Exception(f"Tensor {key} not found in model")


def all_keys(model: dict, index_: dict):
    global file_map
    all_keys_name = []
    if index_ is not None and isinstance(index_, dict) and "weight_map" in index_.keys():
        json_pwd = os.path.dirname(args.input_model.name)
        for (key, val) in index_["weight_map"].items():
            all_keys_name.append(key)
            if val is not None and val not in file_map.keys():
                # JOIN PATH
                val_path = os.path.join(json_pwd, val)
                print(val_path)
                if args.type == "torch":
                    file_map[val] = torch.load(val_path, weights_only=True)
                else:
                    file_map[val] = safe_open(val_path, framework="pt")
    else:
        for key in model.keys():
            if not key.startswith("_"):
                if args.type == "torch":
                    val = model[key]
                if args.type == "safetensor":
                    val = model.get_tensor(key)
                if isinstance(val, torch.Tensor):
                    all_keys_name.append(key)
                elif isinstance(val, dict):
                    all_keys_name.extend(all_keys(val))
                else:
                    pass
    return all_keys_name


def process_str(name: str, type: str='dense'):
    if type == 'dense' or ('down_proj.weight' not in name):
        return name
    return name.replace('weight', 'weight_T')

def process(name: str, ten: torch.Tensor, type: str='dense'):
    if type == 'dense' or ('down_proj.weight' not in name):
        return name, ten

    new_name = name.replace('weight', 'weight_T')
    transposed_tensor = ten.transpose(-2, -1).contiguous()
    return new_name, transposed_tensor


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_model", type=argparse.FileType("r"), default="model.bin"
    )
    parser.add_argument("--output_model", type=str, default="model.mllm")
    parser.add_argument(
        "--type",
        choices=["torch", "safetensor"],
        default="torch",
    )
    parser.add_argument(
        "--model_type",
        choices=["dense", "sparse"],
        default="dense",
    )
    model = None
    index_ = None
    args = parser.parse_args()
    if args.type == "torch":
        if args.input_model.name.endswith(".json"):
            if os.path.basename(args.input_model.name) != "pytorch_model.bin.index.json":
                raise Exception("Only support pytorch_model.bin.index.json")
            index_ = json.load(args.input_model)
        else:
            model = torch.load(args.input_model.name)
            if isinstance(model, dict) and "model" in model.keys():
                model = model["model"]
    elif args.type == "safetensor":
        from safetensors import safe_open

        if args.input_model.name.endswith(".json"):
            index_ = json.load(args.input_model)
        else:
            tensors = {}
            args.input_model.close()
            model = safe_open(args.input_model.name, framework="pt")
            for key in model.keys():
                tensors[key] = model.get_tensor(key)
    else:
        raise Exception("Unknown type")
    writer = Writer(args.output_model)
    model_keys = all_keys(model, index_)
    writer.write_tensor_index_padding([process_str(name, args.model_type) for name in model_keys])

    for key in model_keys:
        tensor = get_tensor(model, key, index_)
        key, tensor = process(key, tensor, args.model_type)
        if tensor.dtype != torch.bool or tensor.dtype != torch.int8:
            tensor = tensor.float()
        offset, size = writer.write_tensor(tensor, key)
        print(f"Get tensor {key} to {offset} with size {size}")

    writer.write_tensor_index()
