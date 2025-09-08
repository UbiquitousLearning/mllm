# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""
Implementation of MLLM Model File V2 format load and store functionality.
This module provides functions to load and save MLLM V2 model files using Python's
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

# MLLM V2 constants
MLLM_MODEL_FILE_V2_MAGIC_NUMBER = 0x519A
MLLM_MODEL_FILE_V2_VERSION = 2
MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH = 512
MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH = 256
MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH = 16


def load_model_file_v2(file_path: str) -> ParamsDict:
    """
    Load an MLLM V2 model file into a ParamsDict.

    Args:
        file_path: Path to the MLLM V2 model file

    Returns:
        ParamsDict containing the model parameters

    Raises:
        ValueError: If the file is not a valid MLLM V2 model file
        IOError: If there are issues reading the file
    """
    try:
        with open(file_path, 'rb') as f:
            # Read header (532 bytes)
            header_data = f.read(532)  # sizeof(ModelFileV2Descriptor) = 532
            if len(header_data) < 532:
                raise ValueError("File too short to be a valid MLLM V2 model file")

            # Unpack header
            magic_number, version, model_name_bytes, num_params, params_desc_offset = struct.unpack(
                '<ii512sIQ', header_data)
            
            # Process model name (null-terminated string)
            model_name = model_name_bytes.split(b'\x00')[0].decode('utf-8')
            
            if magic_number != MLLM_MODEL_FILE_V2_MAGIC_NUMBER:
                raise ValueError(f"Invalid magic number. Expected {MLLM_MODEL_FILE_V2_MAGIC_NUMBER}, got {magic_number}")
            
            if version != MLLM_MODEL_FILE_V2_VERSION:
                raise ValueError(f"Unsupported version. Expected {MLLM_MODEL_FILE_V2_VERSION}, got {version}")

            # Seek to parameter descriptors
            f.seek(params_desc_offset)
            
            # Read parameter descriptors
            params_dict = ParamsDict()
            
            for i in range(num_params):
                # Read one parameter descriptor (352 bytes)
                param_desc_data = f.read(352)  # sizeof(ModelFileV2ParamsDescriptor) = 352
                if len(param_desc_data) < 352:
                    raise ValueError(f"File too short to contain parameter descriptor {i}")

                # Unpack parameter descriptor
                (parameter_id, parameter_type, parameter_size, parameter_offset, 
                 shape_len) = struct.unpack('<IIQQQ', param_desc_data[:36])
                
                # Read shape (128 bytes)
                shape_data = param_desc_data[36:164]  # 32*4 bytes
                shape = list(struct.unpack('<32i', shape_data))[:shape_len]
                
                # Read name (256 bytes)
                name_bytes = param_desc_data[164:420]  # 256 bytes
                name = name_bytes.split(b'\x00')[0].decode('utf-8')

                # Seek to tensor data and read it
                f.seek(parameter_offset)
                data_bytes = f.read(parameter_size)
                if len(data_bytes) < parameter_size:
                    raise IOError(f"Failed to read tensor data for parameter '{name}'")

                # Convert dtype to numpy dtype
                numpy_dtype = _mllm_dtype_to_numpy(parameter_type)
                
                # Calculate number of elements
                bytes_per_element = np.dtype(numpy_dtype).itemsize
                if bytes_per_element == 0:
                    raise ValueError(f"Unsupported data type for parameter '{name}': {parameter_type}")
                
                num_elements = parameter_size // bytes_per_element
                
                # Create numpy array from bytes
                tensor_data = np.frombuffer(data_bytes, dtype=numpy_dtype)
                
                # Reshape according to stored shape
                if shape_len > 0:
                    try:
                        tensor_data = tensor_data.reshape(shape)
                    except ValueError:
                        # Fallback to 1D if reshape fails
                        tensor_data = tensor_data.reshape((num_elements,))
                else:
                    tensor_data = tensor_data.reshape((num_elements,))
                
                # Add to params dict
                params_dict[name] = tensor_data

            return params_dict

    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error loading MLLM V2 model file: {str(e)}")


def store_model_file_v2(params_dict: ParamsDict, file_path: str, model_name: str = "") -> None:
    """
    Store a ParamsDict to an MLLM V2 model file.

    Args:
        params_dict: ParamsDict containing the model parameters
        file_path: Path to save the MLLM V2 model file
        model_name: Optional model name to store in the file header

    Raises:
        IOError: If there are issues writing the file
        ValueError: If there are unsupported tensor types
    """
    try:
        with open(file_path, 'wb') as f:
            num_params = len(params_dict)
            
            # Calculate offsets
            header_size = 532  # sizeof(ModelFileV2Descriptor)
            param_desc_size = 352  # sizeof(ModelFileV2ParamsDescriptor)
            param_desc_total_size = num_params * param_desc_size
            params_desc_offset = header_size  # Right after header
            
            # Calculate where tensor data will start
            tensor_data_start_offset = params_desc_offset + param_desc_total_size
            
            # Write header
            model_name_bytes = model_name.encode('utf-8')
            if len(model_name_bytes) > MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH - 1:
                model_name_bytes = model_name_bytes[:MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH - 1]
            model_name_padded = model_name_bytes.ljust(MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH, b'\x00')
            
            header = struct.pack('<ii512sIQ',
                               MLLM_MODEL_FILE_V2_MAGIC_NUMBER,
                               MLLM_MODEL_FILE_V2_VERSION,
                               model_name_padded,
                               num_params,
                               params_desc_offset)
            f.write(header)

            # Prepare parameter descriptors and data blocks
            param_descriptors = bytearray()
            data_blocks = []
            current_data_offset = tensor_data_start_offset
            
            # Generate parameter descriptors
            for i, (name, tensor) in enumerate(params_dict.items()):
                if _is_supported_tensor(tensor):
                    # Get tensor data and type information
                    tensor_data, mllm_dtype = _get_tensor_data_and_type(tensor)
                    
                    # Parameter ID
                    parameter_id = i
                    
                    # Parameter type
                    parameter_type = mllm_dtype
                    
                    # Parameter size (bytes)
                    parameter_size = len(tensor_data)
                    
                    # Parameter offset
                    parameter_offset = current_data_offset
                    
                    # Shape information
                    if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
                        shape = list(tensor.shape)
                    else:
                        shape = list(tensor.shape)
                    shape_len = len(shape)
                    
                    # Pad shape to fixed length
                    if len(shape) > MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH:
                        raise ValueError(f"Tensor shape too long for parameter '{name}': {len(shape)}")
                    shape_padded = shape + [0] * (MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH - len(shape))
                    
                    # Name
                    name_bytes = name.encode('utf-8')
                    if len(name_bytes) > MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH - 1:
                        name_bytes = name_bytes[:MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH - 1]
                    name_padded = name_bytes.ljust(MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH, b'\x00')

                    # Create parameter descriptor
                    param_desc = struct.pack('<IIQQQ',
                                           parameter_id,
                                           parameter_type,
                                           parameter_size,
                                           parameter_offset,
                                           shape_len)
                    
                    # Add shape data
                    shape_data = struct.pack('<32i', *shape_padded)
                    param_desc += shape_data
                    
                    # Add name
                    param_desc += name_padded
                    
                    param_descriptors.extend(param_desc)
                    
                    # Store data block
                    data_blocks.append(tensor_data)
                    
                    # Update offset for next parameter
                    current_data_offset += parameter_size
                else:
                    raise ValueError(f"Unsupported tensor type for parameter '{name}': {type(tensor)}")
            
            # Write parameter descriptors
            f.write(param_descriptors)
            
            # Write tensor data
            for data_block in data_blocks:
                f.write(data_block)

    except Exception as e:
        raise IOError(f"Error storing MLLM V2 model file: {str(e)}")


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