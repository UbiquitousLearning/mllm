import os
import struct
from typing import BinaryIO

import torch
import platform

def change_dir():

    dir_name = os.path.basename(os.getcwd())
    arch = platform.machine()
    # Check if running on ARM or X86
    if arch == 'aarch64' or 'arm' in arch:
        if dir_name != 'bin-arm':
            os.chdir('../bin-arm')
    else:
        if dir_name != 'bin':
            os.chdir('../bin')





class TestIO:
    file: BinaryIO

    def __init__(self, filename: str, read_mode: bool):
        self.filename = filename
        mode = 'rb' if read_mode else 'wb'
        change_dir()
        self.file = open(f"test_{filename}.mllm", mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    def write_string(self, string: str):
        name = string.encode('utf-8')
        self.file.write(struct.pack("<i", len(name)))
        print("Len:", len(name))
        # print(string.encode('utf-8'))
        self.file.write(name)

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

    def write_longdim(self, n: int, c: int, t: int, h: int, w: int):
        self.file.write(struct.pack("<iiiii", n, c, t, h, w))


class TestSaver(TestIO):
    def __init__(self, filename: str, ):
        super().__init__(filename, False)
        self.write_int(2233)

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

    def write_tensor(self, tensor: torch.Tensor, name: str):
        self.write_string(name)
        self.write_int(self.__torch_dtype_to_int(tensor.dtype))
        dims = list(tensor.shape)
        if len(dims) <= 4:
            if len(dims) > 4:
                raise Exception("Tensor dims should be less than 4")
            dims = [1] * (4 - len(dims)) + dims
            self.write_dim(*dims)
        else:
            dims = dims + [1] * (5 - len(dims))
            self.write_longdim(*dims)
        self.write_u64(0)
        offset = self.file.tell()
        with torch.no_grad():
            tensor.numpy().tofile(self.file)
        end = self.file.tell()
        print(end - offset)
        print(offset)
        self.file.seek(offset - 8)
        self.write_u64(end - offset)
        self.file.seek(end)

    # # One Tensor Index Item Contains: Name_Len(Int)+Name(str)+Int[4]+Weights_Len(UInt64)+Offset(UInt64)+DataType(Int)
    # def calc_tensors_index_table_size(name: str):
    #     return 4 + len(name) + 4 * 4 + 8 + 8 + 4

    # def write_tensor_index(
    #         self,
    # ):
    #     self.file.seek(4 + 8)
    #     for tensor_name in self.tensors_name:
    #         tensor = self.tensors[tensor_name]
    #         # self.write_int(len(tensor.name))
    #         self.write_string(tensor.name)
    #         self.write_int(tensor.dtype)
    #         self.write_dim(*tensor.dims)
    #         self.write_u64(tensor.size)
    #         self.write_u64(tensor.offset)
    #
    # def write_tensor_index_padding(self, tensors_name: list[str]):
    #     if len(tensors_name) > 0:
    #         self.tensors_name = tensors_name
    #         padding_size = reduce(
    #             lambda x, y: x + y, map(self.calc_tensors_index_table_size, tensors_name)
    #         )
    #         self.writer.seek(4)
    #         self.write_u64(padding_size)
    #         print(f"Padding size: {padding_size}")
    #         self.writer.write(b"\x00" * padding_size)
    #         self.writer.flush()
    #         return
    #     else:
    #         raise Exception("No tensors to write")


class TestBase:
    tensors_map: dict[str, torch.Tensor]

    def __init__(self):
        print(self.__class__.__name__)
        self.saver = TestSaver(self.__class__.__name__)
        self.tensors_map = {}

    def add_tensor(self, tensor: torch.Tensor, name: str = None):
        if name is None:
            name = tensor.name
        print("Add", name)
        self.tensors_map[name] = tensor

    def save(self):
        for name, tensor in self.tensors_map.items():
            self.saver.write_tensor(tensor, name)
        self.saver.file.flush()
        self.saver.file.close()

    def test(self):
        pass

    def test_done(self, captrue_tensors: bool = False):
        if captrue_tensors:
            import inspect
            frame = inspect.currentframe()
            try:
                local_ = frame.f_back.f_locals
                for key, value in local_.items():
                    if isinstance(value, torch.Tensor):
                        self.add_tensor(value, key)
            finally:
                del frame
        self.save()
