# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import os
import struct
from typing import List, Union, Dict
from ..ffi import (
    Tensor,
    MLLM_FIND_NUMPY_AVAILABLE,
    MLLM_FIND_TORCH_AVAILABLE,
)

if MLLM_FIND_TORCH_AVAILABLE:
    import torch
if MLLM_FIND_NUMPY_AVAILABLE:
    import numpy as np
from .mllm_type_mapping import MLLM_TYPE_MAPPING


MLLM_MODEL_FILE_V2_MAGIC_NUMBER = 0x519A
MLLM_MODEL_FILE_V2_VERSION = 2
MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH = 512
MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH = 256
MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH = 16


class ModelFileV2Descriptor:
    SIZE = 532

    def __init__(self, model_name: str, num_params: int):
        self.magic = MLLM_MODEL_FILE_V2_MAGIC_NUMBER
        self.version = MLLM_MODEL_FILE_V2_VERSION
        self.model_name = model_name
        self.num_params = num_params
        self.params_desc_offset = 0


class ModelFileV2ParamsDescriptor:
    SIZE = 352

    def __init__(
        self,
        param_id: int,
        param_type: int,
        param_size: int,
        param_offset: int,
        shape: List[int],
        name: str,
    ):
        self.param_id = param_id
        self.param_type = param_type
        self.param_size = param_size
        self.param_offset = param_offset
        self.shape_len = len(shape)
        self.shape = shape
        self.name = name


def _pack_descriptor(desc) -> bytes:
    if isinstance(desc, ModelFileV2Descriptor):
        name_bytes = desc.model_name.encode("utf-8")
        name_bytes = name_bytes.ljust(MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH, b"\0")[
            :MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH
        ]
        return struct.pack(
            f"<II{MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH}sIQ",
            desc.magic,
            desc.version,
            name_bytes,
            desc.num_params,
            desc.params_desc_offset,
        )
    elif isinstance(desc, ModelFileV2ParamsDescriptor):
        name_bytes = desc.name.encode("utf-8")
        name_bytes = name_bytes.ljust(MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH, b"\0")[
            :MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH
        ]
        shape_padded = desc.shape + [0] * (
            MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH - len(desc.shape)
        )
        return struct.pack(
            f"<IIQQQ{MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH}i{MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH}s",
            desc.param_id,
            desc.param_type,
            desc.param_size,
            desc.param_offset,
            desc.shape_len,
            *shape_padded,
            name_bytes,
        )
    else:
        raise TypeError("unknown descriptor type")


class ModelFileV2:
    def __init__(self, file_path, model_name, update_mode="Static", **kwargs):
        self.file_path = file_path
        self.model_name = model_name
        assert update_mode in ["Static", "Streaming"]
        self.update_mode = update_mode
        if update_mode == "Streaming":
            self.max_params_descriptor_buffer_num = kwargs.get(
                "max_params_descriptor_buffer_num", 1024
            )
        self.v2_param_descriptor: List[ModelFileV2ParamsDescriptor] = []
        self.v2_file_header = ModelFileV2Descriptor(
            model_name="",
            num_params=0
            if update_mode == "Static"
            else self.max_params_descriptor_buffer_num,
        )

        # Open file
        self.file_handler = open(self.file_path, "wb")

        # If mode is Streaming. We need to occupy the necessary space for params descriptor and header.
        if update_mode == "Streaming":
            # Write empty 0s to sizeof(ModelFileV2Descriptor) and self.max_params_descriptor_buffer_num * sizeof(ModelFileV2ParamsDescriptor)
            header_size = ModelFileV2Descriptor.SIZE
            param_desc_size = ModelFileV2ParamsDescriptor.SIZE
            total_param_desc_size = (
                self.max_params_descriptor_buffer_num * param_desc_size
            )

            # Reserve space: header + parameter descriptors
            self.v2_file_header.params_desc_offset = header_size
            reserved_bytes = b"\x00" * (header_size + total_param_desc_size)
            self.file_handler.write(reserved_bytes)
            self.file_handler.flush()

    def streaming_write(self, tensor_name, tensor_obj):
        if MLLM_FIND_TORCH_AVAILABLE and isinstance(tensor_obj, torch.Tensor):
            # PyTorch tensor
            shape = list(tensor_obj.shape)
            tensor_data = tensor_obj.detach().cpu().numpy().tobytes()
            true_dtype = MLLM_TYPE_MAPPING[tensor_obj.dtype]
        elif MLLM_FIND_NUMPY_AVAILABLE and isinstance(tensor_obj, np.ndarray):
            # Numpy array
            shape = list(tensor_obj.shape)
            tensor_data = tensor_obj.tobytes()
            true_dtype = MLLM_TYPE_MAPPING[tensor_obj.dtype]
        elif isinstance(tensor_obj, Tensor):
            # Mllm Tensor
            shape = list(tensor_obj.shape)
            tensor_data = tensor_obj.tobytes()
            true_dtype = tensor_obj.dtype.to_pod()
        else:
            raise TypeError(
                "Unsupported tensor type. Only torch.Tensor and np.ndarray are supported."
            )

        tensor_size = len(tensor_data)

        assert len(self.v2_param_descriptor) <= self.max_params_descriptor_buffer_num
        desc = ModelFileV2ParamsDescriptor(
            param_id=len(self.v2_param_descriptor),
            param_type=true_dtype,
            param_size=tensor_size,
            param_offset=self.file_handler.tell(),
            shape=shape,
            name=tensor_name,
        )

        self.v2_param_descriptor.append(desc)

        # Write at tail
        self.file_handler.write(tensor_data)

        # Update num_params
        self.v2_file_header.num_params = len(self.v2_param_descriptor)
        desc_position = self.v2_file_header.params_desc_offset + (
            desc.param_id * ModelFileV2ParamsDescriptor.SIZE
        )

        # Find the updated desc_position and write data into it.
        self.file_handler.seek(desc_position)
        self.file_handler.write(_pack_descriptor(desc))

        # Back to tail.
        self.file_handler.seek(0, os.SEEK_END)

    def static_write(self, tensor_obj):
        # Calculate total size needed for parameter descriptors
        total_params = len(tensor_obj)

        # Pre-allocate parameter descriptors list
        self.v2_param_descriptor = [None] * total_params

        # Write header and reserve space for parameter descriptors
        header_size = ModelFileV2Descriptor.SIZE
        param_desc_size = ModelFileV2ParamsDescriptor.SIZE
        total_param_desc_size = total_params * param_desc_size

        # Reserve space: header + parameter descriptors
        self.v2_file_header.params_desc_offset = header_size
        reserved_bytes = b"\x00" * (header_size + total_param_desc_size)
        self.file_handler.write(reserved_bytes)
        self.file_handler.flush()

        # Write tensor data
        param_id = 0
        for tensor_name, tensor in tensor_obj.items():
            if MLLM_FIND_TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                # PyTorch tensor
                shape = list(tensor.shape)
                tensor_data = tensor.detach().cpu().numpy().tobytes()
                true_dtype = MLLM_TYPE_MAPPING[tensor.dtype]
            elif MLLM_FIND_NUMPY_AVAILABLE and isinstance(tensor, np.ndarray):
                # Numpy array
                shape = list(tensor.shape)
                tensor_data = tensor.tobytes()
                true_dtype = MLLM_TYPE_MAPPING[tensor.dtype]
            elif isinstance(tensor, Tensor):
                # Mllm Tensor
                shape = list(tensor.shape)
                tensor_data = tensor.tobytes()
                true_dtype = tensor.dtype.to_pod()
            else:
                raise TypeError(
                    "Unsupported tensor type. Only torch.Tensor, np.ndarray and Tensor are supported."
                )

            tensor_size = len(tensor_data)

            desc = ModelFileV2ParamsDescriptor(
                param_id=param_id,
                param_type=true_dtype,
                param_size=tensor_size,
                param_offset=self.file_handler.tell(),
                shape=shape,
                name=tensor_name,
            )

            self.v2_param_descriptor[param_id] = desc

            # Write at tail
            self.file_handler.write(tensor_data)
            param_id += 1

        # Update num_params
        self.v2_file_header.num_params = len(self.v2_param_descriptor)

        # Write parameter descriptors
        for desc in self.v2_param_descriptor:
            desc_position = self.v2_file_header.params_desc_offset + (
                desc.param_id * ModelFileV2ParamsDescriptor.SIZE
            )
            # Find the updated desc_position and write data into it.
            self.file_handler.seek(desc_position)
            self.file_handler.write(_pack_descriptor(desc))

        # Back to tail.
        self.file_handler.seek(0, os.SEEK_END)

    def finalize(self):
        # Update header
        self.file_handler.seek(0)
        self.file_handler.write(_pack_descriptor(self.v2_file_header))
        self.file_handler.flush()
