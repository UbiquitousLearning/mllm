import argparse
import struct
from functools import reduce
from io import BufferedWriter

import torch

MAGIC_NUMBER = 20012


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
    tensors_map: dict[str, Tensor]
    tensors_name: list[str]

    def __init__(self, path: str):
        self.tensors_map = {}
        self.tensors_name = []
        self.writer = open(path, "wb+")
        self.writer.seek(0)
        self.write_int(MAGIC_NUMBER)

    def __torch_dtype_to_int(self, dtype: torch.dtype) -> int:
        if dtype == torch.float32:
            return 0
        elif dtype == torch.float64:
            return 1
        elif dtype == torch.float16:
            return 2
        elif dtype == torch.uint8:
            return 3
        elif dtype == torch.int8:
            return 4
        elif dtype == torch.int16:
            return 5
        elif dtype == torch.int32:
            return 6
        elif dtype == torch.int64:
            return 7
        elif dtype == torch.bool:
            return 8
        elif dtype == torch.qint8:
            return 9
        elif dtype == torch.quint8:
            return 10
        elif dtype == torch.qint32:
            return 11
        elif dtype == torch.bfloat16:
            return 12
        else:
            raise Exception("Unknown dtype: " + str(dtype))

    def write_int(self, val: int):
        self.writer.write(struct.pack("<i", val))

    def write_float(self, val: float):
        self.writer.write(struct.pack("<f", val))

    def write_u64(self, val: int):
        self.writer.write(struct.pack("<Q", val))

    def write_str(self, val: str):
        self.writer.write(struct.pack("<i", len(val)))
        self.writer.write(val.encode("utf-8"))

    def write_tensor(self, tensor: torch.Tensor, name: str) -> tuple[int, int]:
        tensor_idx = Tensor(name=name, dtype=self.__torch_dtype_to_int(tensor.dtype))
        self.tensors_map[name] = tensor_idx
        offset = self.writer.tell()
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
            self.write_str(tensor.name)
            self.write_u64(tensor.size)
            self.write_u64(tensor.offset)
            self.write_int(tensor.dtype)

    def write_tensor_index_padding(self, tensors_name: list[str]):
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


def get_tensor(model: dict, key: str):
    if key in model.keys():
        if args.type == "torch":
            return model[key]
        if args.type == "safetensor":
            return model.get_tensor(key)
    else:
        raise Exception(f"Tensor {key} not found in model")


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
    args = parser.parse_args()
    if args.type == "torch":
        model = torch.load(args.input_model)
    elif args.type == "safetensor":
        from safetensors import safe_open

        args.input_model.close()
        model = safe_open(args.input_model.name, framework="pt")
    else:
        raise Exception("Unknown type")
    writer = Writer(args.output_model)
    model_keys = [key for key in model.keys() if not key.startswith("_")]
    writer.write_tensor_index_padding(model_keys)
    for key in model_keys:
        tensor = get_tensor(model, key)
        offset, size = writer.write_tensor(tensor, key)
        print(f"Write tensor {key} to {offset} with size {size}")
    writer.write_tensor_index()
