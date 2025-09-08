#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import argparse
import sys
from pathlib import Path

# Add the parent directory to sys.path to import convertor module
sys.path.append(str(Path(__file__).parent.parent))

from convertor import (
    convert_torch_model_to_mllm,
    convert_safetensors_model_to_mllm,
    _detect_mllm_version,
    load_torch_model,
    load_safetensors_model,
    store_model_file_v1,
    store_model_file_v2,
)
from convertor.params_dict import ParamsDict


def cast_params_to_fp32(params_dict: ParamsDict) -> ParamsDict:
    """
    Cast all parameters in the ParamsDict to FP32.

    Args:
        params_dict: ParamsDict containing the model parameters

    Returns:
        ParamsDict with all parameters cast to FP32
    """
    fp32_params_dict = ParamsDict()

    # Try to import required libraries
    try:
        import torch

        TORCH_AVAILABLE = True
    except ImportError:
        TORCH_AVAILABLE = False

    try:
        import numpy as np

        NUMPY_AVAILABLE = True
    except ImportError:
        NUMPY_AVAILABLE = False

    for key, value in params_dict.items():
        if hasattr(value, "dtype"):  # Check if the value has a dtype attribute
            # For PyTorch tensors
            if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                # Cast to FP32 if not already
                if value.dtype != torch.float32:
                    fp32_value = value.to(dtype=torch.float32)
                    fp32_params_dict[key] = fp32_value
                else:
                    fp32_params_dict[key] = value
            # For NumPy arrays
            elif NUMPY_AVAILABLE and isinstance(value, np.ndarray):
                # Cast to FP32 if not already
                if value.dtype != np.float32:
                    fp32_value = value.astype(np.float32)
                    fp32_params_dict[key] = fp32_value
                else:
                    fp32_params_dict[key] = value
            else:
                # If we can't determine how to cast, keep the original value
                fp32_params_dict[key] = value
        else:
            # If the value doesn't have a dtype, keep it as is
            fp32_params_dict[key] = value

    return fp32_params_dict


def convert_model_with_options(
    input_path: str,
    output_path: str,
    format: str,
    model_name: str = "",
    cast_to_fp32: bool = False,
):
    """
    Convert model with various options.

    Args:
        input_path: Input model file path
        output_path: Output MLLM model file path
        format: Output format (mllm-v1 or mllm-v2)
        model_name: Model name (used for MLLM V2 format)
        cast_to_fp32: Whether to cast all parameters to FP32
    """
    # Determine input format based on file extension
    if (
        input_path.endswith(".pth")
        or input_path.endswith(".pt")
        or input_path.endswith(".bin")
    ):
        print(f"Loading PyTorch model from {input_path}...")
        params_dict = load_torch_model(input_path)
    elif input_path.endswith(".safetensors") or input_path.endswith(
        ".safetensors.index.json"
    ):
        print(f"Loading Safetensors model from {input_path}...")
        params_dict = load_safetensors_model(input_path)
    else:
        raise ValueError(
            f"Unsupported input format for file: {input_path}. "
            f"Supported formats are: .pth, .pt, .bin, .safetensors"
        )

    # Cast to FP32 if requested
    if cast_to_fp32:
        print("Casting all parameters to FP32...")
        params_dict = cast_params_to_fp32(params_dict)

    # Store model in the specified MLLM format
    print(f"Converting model to {format} format...")
    if format == "mllm-v1":
        store_model_file_v1(params_dict, output_path)
    elif format == "mllm-v2":
        store_model_file_v2(params_dict, output_path, model_name)

    print(f"Successfully converted model to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MLLM Model Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mllm-convertor --input model.pth --output model.mllm --format mllm-v2
  mllm-convertor --input model.safetensors --output model.mllm --format mllm-v1
  mllm-convertor --input model.pth --output model.mllm --format mllm-v2 --model-name "my-model"
  mllm-convertor --input model.pth --output model.mllm --format mllm-v2 --cast-all-to-fp32
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input model file path (.pth or .safetensors)",
    )

    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output MLLM model file path"
    )

    parser.add_argument(
        "--format",
        "-f",
        type=str,
        choices=["mllm-v1", "mllm-v2"],
        default="mllm-v2",
        help="Output MLLM format version (default: mllm-v2)",
    )

    parser.add_argument(
        "--model-name",
        "-n",
        type=str,
        default="",
        help="Model name (used for MLLM V2 format)",
    )

    parser.add_argument(
        "--cast-all-to-fp32",
        action="store_true",
        help="Cast all parameters to FP32",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    try:
        convert_model_with_options(
            str(input_path),
            str(output_path),
            args.format,
            args.model_name,
            args.cast_all_to_fp32,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
