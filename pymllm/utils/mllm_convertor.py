# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

from .. import convertor
from ..convertor.model_file_v2 import ModelFileV2
from ..quantize.solver import QuantizeSolver
from ..quantize.pipeline import BUILTIN_QUANTIZE_PIPELINE


def main():
    parser = argparse.ArgumentParser(description="MLLM Model Converter")
    parser.add_argument(
        "--input_path", type=str, help="Path to input model file", required=True
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to output model file", required=True
    )
    parser.add_argument("--model_name", type=str, help="Model name", required=True)
    parser.add_argument("--cfg_path", type=str, help="Quantization config file path")
    parser.add_argument(
        "--format",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Output format version (default: v2)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        help=f"Choose builtin pipeline in {BUILTIN_QUANTIZE_PIPELINE.keys()}",
    )
    parser.add_argument("--passes", type=List, help="Passes that performs on params")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    if args.verbose:
        print(f"Converting {args.input_path} to {args.output_path}")
        print(f"Output format: {args.format}")

    # Get params
    params = convertor.load_model(args.input_path)

    # Build pipeline
    if args.cfg_path is None:
        # TODO just convert to mllm file
        pass
    elif (
        args.cfg_path is not None and args.pipeline is not None and args.format == "v2"
    ):
        cfg = None
        with open(args.cfg_path) as f:
            cfg = json.load(f)
        pipeline: QuantizeSolver = BUILTIN_QUANTIZE_PIPELINE[args.pipeline]()

        old_param_size = len(params)
        new_param_size = pipeline.stream_quantize_params_size(cfg, params)

        print(f"Params Num: Before: {old_param_size}, After: {new_param_size}")

        pipeline.stream_quantize(
            cfg,
            params,
            writer=ModelFileV2(
                args.output_path,
                args.model_name,
                "Streaming",
                max_params_descriptor_buffer_num=new_param_size,
            ),
            cast_left_2_fp32=True,
            verbose=args.verbose,
        )
