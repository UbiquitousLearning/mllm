# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""
Implementation of MLLM Model File V1 format load and store functionality.
This module provides functions to load and save MLLM V1 model files using Python's
serialization tools without C++ bindings.
"""

import struct
import numpy as np
from typing import Dict, Any
from .params_dict import ParamsDict
from .mllm_type_mapping import MLLM_TYPE_MAPPING

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# MLLM V1 constants
MLLM_MODEL_FILE_V1_MAGIC_NUMBER = 20012


def load_model_file_v1(file_path: str) -> ParamsDict:
    """
    Load an MLLM V1 model file into a ParamsDict.

    Args:
        file_path: Path to the MLLM V1 model file

    Returns:
        ParamsDict containing the model parameters

    Raises:
        ValueError: If the file is not a valid MLLM V1 model file
        IOError: If there are issues reading the file
    """
    try:
        with open(file_path, 'rb') as f:
            # Read header
            header_data = f.read(12)  # sizeof(ModelFileV1Descriptor) = 12
            if len(header_data) < 12:
                raise ValueError("File too short to be a valid MLLM V1 model file")

            magic_number, parameter_desc_offset = struct.unpack('<iQ', header_data)
            
            if magic_number != MLLM_MODEL_FILE_V1_MAGIC_NUMBER:
                raise ValueError(f"Invalid magic number. Expected {MLLM_MODEL_FILE_V1_MAGIC_NUMBER}, got {magic_number}")

            # Read parameter descriptors
            descriptors_data = f.read(parameter_desc_offset)
            if len(descriptors_data) < parameter_desc_offset:
                raise ValueError("File too short to contain expected parameter descriptors")

            # Parse parameter descriptors
            params_dict = ParamsDict()
            pos = 0
            current_data_offset = 12 + parameter_desc_offset  # header + descriptors

            while pos < len(descriptors_data):
                # Read name length
                if pos + 4 > len(descriptors_data):
                    break
                name_len = struct.unpack_from('<I', descriptors_data, pos)[0]
                pos += 4

                # Read name
                if pos + name_len > len(descriptors_data):
                    break
                name = descriptors_data[pos:pos + name_len].decode('utf-8')
                pos += name_len

                # Read data length
                if pos + 8 > len(descriptors_data):
                    break
                data_len = struct.unpack_from('<Q', descriptors_data, pos)[0]
                pos += 8

                # Read offset
                if pos + 8 > len(descriptors_data):
                    break
                offset = struct.unpack_from('<Q', descriptors_data, pos)[0]
                pos += 8

                # Read data type
                if pos + 4 > len(descriptors_data):
                    break
                dtype = struct.unpack_from('<i', descriptors_data, pos)[0]
                pos += 4

                # Seek to data position and read data
                f.seek(offset)
                data_bytes = f.read(data_len)
                if len(data_bytes) < data_len:
                    raise IOError(f"Failed to read tensor data for parameter '{name}'")

                # Convert dtype to numpy dtype
                numpy_dtype = _mllm_dtype_to_numpy(dtype)
                
                # Create numpy array from bytes
                # Calculate number of elements based on dtype
                bytes_per_element = np.dtype(numpy_dtype).itemsize
                num_elements = data_len // bytes_per_element
                
                # Create array and reshape to 1D for now
                tensor_data = np.frombuffer(data_bytes, dtype=numpy_dtype)
                tensor_data = tensor_data.reshape((num_elements,))
                
                # Add to params dict
                params_dict[name] = tensor_data

                # Update expected data offset for next parameter
                current_data_offset = offset + data_len

            return params_dict

    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error loading MLLM V1 model file: {str(e)}")


def store_model_file_v1(params_dict: ParamsDict, file_path: str) -> None:
    """
    Store a ParamsDict to an MLLM V1 model file.

    Args:
        params_dict: ParamsDict containing the model parameters
        file_path: Path to save the MLLM V1 model file

    Raises:
        IOError: If there are issues writing the file
        ValueError: If there are unsupported tensor types
    """
    try:
        with open(file_path, 'wb') as f:
            # Calculate header size and parameter descriptors size
            header_size = 12  # sizeof(ModelFileV1Descriptor)
            param_desc_total_size = 0

            # Calculate total size of the descriptor section
            for name, tensor in params_dict.items():
                if _is_supported_tensor(tensor):
                    param_desc_total_size += 4  # name length (uint32_t)
                    param_desc_total_size += len(name)  # name string
                    param_desc_total_size += 8  # data length (uint64_t)
                    param_desc_total_size += 8  # offset (uint64_t)
                    param_desc_total_size += 4  # data type (int32_t)
                else:
                    raise ValueError(f"Unsupported tensor type for parameter '{name}': {type(tensor)}")

            # Write header
            parameter_desc_offset = param_desc_total_size
            header = struct.pack('<iQ', MLLM_MODEL_FILE_V1_MAGIC_NUMBER, parameter_desc_offset)
            f.write(header)

            # Write parameter descriptors and calculate data offsets
            current_data_offset = header_size + param_desc_total_size
            
            # First pass: write descriptors
            descriptors_data = bytearray()
            data_blocks = []  # Store data blocks to write later
            
            for name, tensor in params_dict.items():
                if _is_supported_tensor(tensor):
                    # Get tensor data and type information
                    tensor_data, mllm_dtype = _get_tensor_data_and_type(tensor)
                    
                    # Name length
                    name_len = len(name)
                    descriptors_data.extend(struct.pack('<I', name_len))
                    
                    # Name
                    descriptors_data.extend(name.encode('utf-8'))
                    
                    # Data length (bytes)
                    data_len = len(tensor_data)
                    descriptors_data.extend(struct.pack('<Q', data_len))
                    
                    # Offset
                    descriptors_data.extend(struct.pack('<Q', current_data_offset))
                    
                    # Data type
                    descriptors_data.extend(struct.pack('<i', mllm_dtype))
                    
                    # Store data block for later writing
                    data_blocks.append(tensor_data)
                    
                    # Update offset for next parameter
                    current_data_offset += data_len
                else:
                    raise ValueError(f"Unsupported tensor type for parameter '{name}': {type(tensor)}")
            
            # Write descriptors
            f.write(descriptors_data)
            
            # Write tensor data
            for data_block in data_blocks:
                f.write(data_block)

    except Exception as e:
        raise IOError(f"Error storing MLLM V1 model file: {str(e)}")


def _mllm_dtype_to_numpy(dtype: int) -> np.dtype:
    """
    Convert MLLM data type to NumPy data type.

    Args:
        dtype: MLLM data type ID

    Returns:
        Corresponding NumPy data type
    """
    # Create reverse mapping from MLLM types to NumPy types
    mllm_to_numpy = {}
    for k, v in MLLM_TYPE_MAPPING.items():
        # Only consider numpy types
        if isinstance(k, type(np.float32)):  
            mllm_to_numpy[v] = k
    
    # Special handling for bfloat16 - map to float32 as numpy doesn't support bfloat16 directly
    if dtype == 128:  # kBFloat16
        return np.float32
    
    if dtype in mllm_to_numpy:
        return mllm_to_numpy[dtype]
    
    # Default fallback
    return np.float32


def _numpy_dtype_to_mllm(dtype: np.dtype) -> int:
    """
    Convert NumPy data type to MLLM data type.

    Args:
        dtype: NumPy data type

    Returns:
        Corresponding MLLM data type ID
    """
    # First try direct mapping
    if dtype in MLLM_TYPE_MAPPING:
        return MLLM_TYPE_MAPPING[dtype]
    
    # Try with PyTorch if available
    if TORCH_AVAILABLE:
        # Try to find a PyTorch equivalent
        numpy_to_torch = {
            np.float32: torch.float32,
            np.float16: torch.float16,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
        }
        
        if dtype in numpy_to_torch:
            torch_dtype = numpy_to_torch[dtype]
            if torch_dtype in MLLM_TYPE_MAPPING:
                return MLLM_TYPE_MAPPING[torch_dtype]
    
    # Default to float32
    if np.float32 in MLLM_TYPE_MAPPING:
        return MLLM_TYPE_MAPPING[np.float32]
    
    # Ultimate fallback
    return 0


def _get_tensor_data_and_type(tensor):
    """
    Get raw tensor data bytes and corresponding MLLM data type.
    
    Args:
        tensor: Input tensor (numpy array or PyTorch tensor)
        
    Returns:
        tuple: (raw_data_bytes, mllm_dtype)
    """
    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
        # For PyTorch tensors, get raw bytes directly
        if tensor.dtype == torch.bfloat16:
            # PyTorch bfloat16 tensors can be directly converted to bytes
            mllm_dtype = 128  # kBFloat16
            raw_data = tensor.detach().cpu().view(torch.uint8).numpy().tobytes()
            return raw_data, mllm_dtype
        else:
            # For other PyTorch tensors, convert to numpy first
            numpy_tensor = tensor.detach().cpu().numpy()
            mllm_dtype = _numpy_dtype_to_mllm(numpy_tensor.dtype)
            raw_data = numpy_tensor.tobytes()
            return raw_data, mllm_dtype
    else:
        # For numpy arrays
        mllm_dtype = _numpy_dtype_to_mllm(tensor.dtype)
        raw_data = tensor.tobytes()
        return raw_data, mllm_dtype


def _is_supported_tensor(tensor) -> bool:
    """
    Check if a tensor is of a supported type.

    Args:
        tensor: The tensor to check

    Returns:
        True if the tensor is supported, False otherwise
    """
    return isinstance(tensor, np.ndarray) or (TORCH_AVAILABLE and isinstance(tensor, torch.Tensor))