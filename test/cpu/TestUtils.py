import struct
from functools import reduce
from typing import BinaryIO

import torch


class TestIO:
    file: BinaryIO

    def __init__(self, filename: str, read_mode: bool):
        self.filename = filename
        mode = 'rb' if read_mode else 'wb'
        self.file = open(filename, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def write_string(self, string: str):
        self.file.write(struct.pack("<i", len(string)))
        self.file.write(string.encode('utf-8'))

    def write_int(self, integer: int):
        self.file.write(struct.pack("<i", integer))

    def write_float(self, float: float):
        self.file.write(struct.pack("<d", float))

    def write_u64(self, u64: int):
        self.file.write(struct.pack("<Q", u64))

    def write(self, data):
        self.file.write(data)

    def write_dim(self, n: int, c: int, h: int, w: int):
        self.file.write(struct.pack("<iiii", n, c, h, w))


class Tensor:
    name: str
    offset: int
    size: int
    dtype: int
    dims: list[int]

    def __init__(self, name: str, dtype: int):
        self.name = name
        self.dtype = dtype


class TestSaver(TestIO):
    tensors: dict[str, Tensor]

    def __init__(self, filename: str, ):
        super().__init__(filename, False)
        self.tensors = {}

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

    def write_tensor(self, tensor: torch.Tensor, name: str) -> tuple[int, int]:
        tensor_idx = Tensor(name=name, dtype=self.__torch_dtype_to_int(tensor.dtype))
        self.tensors[name] = tensor_idx
        offset = self.file.tell()
        dims = list(tensor.shape)
        if len(dims) > 4:
            raise Exception("Tensor dims should be less than 4")
        dims = dims + [1] * (4 - len(dims))
        tensor_idx.dims = dims
        tensor_numpy = tensor.numpy()
        tensor_numpy.tofile(self.file)
        size = self.file.tell() - offset
        tensor_idx.size = size
        tensor_idx.offset = offset
        return offset, size

    # One Tensor Index Item Contains: Name_Len(Int)+Name(str)+Int[4]+Weights_Len(UInt64)+Offset(UInt64)+DataType(Int)
    def calc_tensors_index_table_size(name: str):
        return 4 + len(name) + 4 * 4 + 8 + 8 + 4

    def write_tensor_index(
            self,
    ):
        self.file.seek(4 + 8)
        for tensor_name in self.tensors_name:
            tensor = self.tensors[tensor_name]
            # self.write_int(len(tensor.name))
            self.write_string(tensor.name)
            self.write_int(tensor.dtype)
            self.write_dim(*tensor.dims)
            self.write_u64(tensor.size)
            self.write_u64(tensor.offset)

    def write_tensor_index_padding(self, tensors_name: list[str]):
        if len(tensors_name) > 0:
            self.tensors_name = tensors_name
            padding_size = reduce(
                lambda x, y: x + y, map(self.calc_tensors_index_table_size, tensors_name)
            )
            self.writer.seek(4)
            self.write_u64(padding_size)
            print(f"Padding size: {padding_size}")
            self.writer.write(b"\x00" * padding_size)
            self.writer.flush()
            return
        else:
            raise Exception("No tensors to write")
