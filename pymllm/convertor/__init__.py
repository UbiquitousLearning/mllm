# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""
Unified interface for loading and storing models in various formats.
"""

from .params_dict import ParamsDict
from .model_file_v1 import load_model_file_v1, store_model_file_v1
from .model_file_v2 import load_model_file_v2, store_model_file_v2
from .params_dict import load_torch_model, load_safetensors_model
import struct

# Re-export key classes and functions
__all__ = [
    "ParamsDict",
    "load_model",
    "store_model",
    "load_torch_model",
    "load_safetensors_model",
    "convert_torch_model_to_mllm",
    "convert_safetensors_model_to_mllm",
    "_detect_mllm_version"
]


def load_model(file_path: str, format: str = None) -> ParamsDict:
    """
    Load a model from file into a ParamsDict.

    Args:
        file_path: Path to the model file
        format: Format of the model file. If None, infer from file extension or content.
                Supported formats: 'mllm-v1', 'mllm-v2', 'torch', 'safetensors'

    Returns:
        ParamsDict containing the model parameters

    Raises:
        ValueError: If the format is not supported or cannot be inferred
        IOError: If there are issues reading the file
    """
    if format is None:
        # Infer format from file extension or content
        if file_path.endswith('.pt') or file_path.endswith('.pth') or file_path.endswith('.bin'):
            format = 'torch'
        elif file_path.endswith('.safetensors') or file_path.endswith('.safetensors.index.json'):
            format = 'safetensors'
        elif file_path.endswith('.mllm'):
            # For .mllm files, we need to check the header to determine version
            format = _detect_mllm_version(file_path)
        else:
            # Try to detect MLLM version by header
            try:
                format = _detect_mllm_version(file_path)
            except:
                raise ValueError(f"Cannot infer model format from file path: {file_path}")

    if format == 'mllm-v1':
        return load_model_file_v1(file_path)
    elif format == 'mllm-v2':
        return load_model_file_v2(file_path)
    elif format == 'torch':
        return load_torch_model(file_path)
    elif format == 'safetensors':
        return load_safetensors_model(file_path)
    else:
        raise ValueError(f"Unsupported model format: {format}")


def store_model(params_dict: ParamsDict, file_path: str, format: str,
                model_name: str = "") -> None:
    """
    Store a ParamsDict to a model file.

    Args:
        params_dict: ParamsDict containing the model parameters
        file_path: Path to save the model file
        format: Format of the model file.
                Supported formats: 'mllm-v1', 'mllm-v2', 'torch', 'safetensors'
        model_name: Optional model name (used for MLLM formats)

    Raises:
        ValueError: If the format is not supported
        IOError: If there are issues writing the file
    """
    if format == 'mllm-v1':
        store_model_file_v1(params_dict, file_path)
    elif format == 'mllm-v2':
        store_model_file_v2(params_dict, file_path, model_name)
    elif format == 'torch':
        params_dict.save_pytorch(file_path)
    elif format == 'safetensors':
        params_dict.save_safetensors(file_path)
    else:
        raise ValueError(f"Unsupported model format: {format}")


def convert_torch_model_to_mllm(torch_model_path: str, mllm_model_path: str, 
                               format: str = 'mllm-v2', model_name: str = "") -> None:
    """
    Convert a PyTorch model directly to MLLM model format.

    Args:
        torch_model_path: Path to the PyTorch model file
        mllm_model_path: Path to save the MLLM model file
        format: Target MLLM format ('mllm-v1' or 'mllm-v2')
        model_name: Optional model name (used for MLLM formats)

    Raises:
        ValueError: If the format is not supported
        IOError: If there are issues reading or writing files
    """
    # Load PyTorch model
    params_dict = load_torch_model(torch_model_path)
    
    # Store as MLLM model
    store_model(params_dict, mllm_model_path, format, model_name)


def convert_safetensors_model_to_mllm(safetensors_model_path: str, mllm_model_path: str,
                                     format: str = 'mllm-v2', model_name: str = "") -> None:
    """
    Convert a Safetensors model directly to MLLM model format.

    Args:
        safetensors_model_path: Path to the Safetensors model file
        mllm_model_path: Path to save the MLLM model file
        format: Target MLLM format ('mllm-v1' or 'mllm-v2')
        model_name: Optional model name (used for MLLM formats)

    Raises:
        ValueError: If the format is not supported
        IOError: If there are issues reading or writing files
    """
    # Load Safetensors model
    params_dict = load_safetensors_model(safetensors_model_path)
    
    # Store as MLLM model
    store_model(params_dict, mllm_model_path, format, model_name)


def _detect_mllm_version(file_path: str) -> str:
    """
    Detect MLLM model version by reading file header.

    Args:
        file_path: Path to the MLLM model file

    Returns:
        'mllm-v1' or 'mllm-v2' based on the file header

    Raises:
        ValueError: If the file is not a valid MLLM model file
    """
    try:
        with open(file_path, 'rb') as f:
            # Try to read as V1 first (12 bytes header)
            header_data = f.read(12)
            if len(header_data) >= 12:
                magic_number, _ = struct.unpack('<iQ', header_data[:12])
                if magic_number == 20012:  # MLLM V1 magic number
                    return 'mllm-v1'
            
            # Try to read as V2 (532 bytes header)
            f.seek(0)
            header_data = f.read(532)
            if len(header_data) >= 532:
                magic_number, version = struct.unpack('<ii', header_data[:8])
                if magic_number == 0x519A and version == 2:  # MLLM V2 magic number and version
                    return 'mllm-v2'
            
            raise ValueError("File is not a valid MLLM model file")
    except Exception:
        raise ValueError("File is not a valid MLLM model file")