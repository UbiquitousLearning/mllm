#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""Validate a converted Ling-3.0-tiny MLLM V2 model without loading tensor data."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path

from validate_checkpoint import expected_shapes, validate_config, validate_recipe


MODEL_HEADER = struct.Struct("<II512sIQ")
PARAMETER_DESCRIPTOR = struct.Struct("<IIQQQ16i256s")
MODEL_MAGIC = 0x519A
MODEL_VERSION = 2
FLOAT32 = 0
BYTE = 134


def packed_size(out_channels: int, in_channels: int) -> int:
    if out_channels <= 0 or in_channels <= 0 or in_channels % 32:
        raise AssertionError("KAI W4A32 shapes require positive dimensions and K divisible by 32")
    nr = 8
    blocks_per_row = in_channels // 32
    bytes_per_nr_rows = nr * (blocks_per_row * (16 + 2) + 4 + 4)
    return math.ceil(out_channels / nr) * bytes_per_nr_rows


def decode_c_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8")


def expected_descriptors(shapes: dict[str, list[int]], recipe: dict) -> dict[str, tuple[int, list[int], int]]:
    expected = {name: (FLOAT32, shape, math.prod(shape) * 4) for name, shape in shapes.items()}
    validate_recipe(shapes, recipe)
    for pattern_text, entry in recipe.items():
        pattern = re.compile(pattern_text)
        for name, shape in shapes.items():
            if pattern.fullmatch(name):
                size = packed_size(shape[0], shape[1])
                expected[name] = (BYTE, [size], size)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--runtime-config", type=Path, default=Path(__file__).with_name("config_tiny_w4a32_kai.json"))
    parser.add_argument("--quant-config", type=Path, default=Path(__file__).with_name("quant_cfg_tiny_w4a32_kai.json"))
    parser.add_argument("--model-name", default="Ling-3.0-tiny")
    args = parser.parse_args()
    source_config = json.loads((args.checkpoint / "config.json").read_text())
    runtime_config = json.loads(args.runtime_config.read_text())
    recipe = json.loads(args.quant_config.read_text())
    validate_config(source_config, runtime_config)
    expected = expected_descriptors(expected_shapes(), recipe)

    file_size = args.model.stat().st_size
    with args.model.open("rb") as model_file:
        raw_header = model_file.read(MODEL_HEADER.size)
        if len(raw_header) != MODEL_HEADER.size:
            raise AssertionError("Truncated MLLM model header")
        magic, version, raw_name, num_parameters, descriptor_offset = MODEL_HEADER.unpack(raw_header)
        identity = (magic, version, decode_c_string(raw_name), descriptor_offset)
        expected_identity = (MODEL_MAGIC, MODEL_VERSION, args.model_name, MODEL_HEADER.size)
        if identity != expected_identity:
            raise AssertionError(f"Invalid model identity: expected {expected_identity}, got {identity}")
        if num_parameters != len(expected):
            raise AssertionError(f"Expected {len(expected)} parameters, model declares {num_parameters}")
        actual: dict[str, tuple[int, list[int], int, int]] = {}
        for expected_id in range(num_parameters):
            raw_descriptor = model_file.read(PARAMETER_DESCRIPTOR.size)
            if len(raw_descriptor) != PARAMETER_DESCRIPTOR.size:
                raise AssertionError(f"Truncated descriptor {expected_id}")
            values = PARAMETER_DESCRIPTOR.unpack(raw_descriptor)
            parameter_id, dtype, size, offset, shape_length = values[:5]
            if parameter_id != expected_id or shape_length > 16:
                raise AssertionError(f"Invalid parameter descriptor {expected_id}")
            name = decode_c_string(values[21])
            if name in actual:
                raise AssertionError(f"Duplicate converted parameter: {name}")
            actual[name] = (dtype, list(values[5:21])[:shape_length], size, offset)

    if set(actual) != set(expected):
        raise AssertionError(
            f"Converted tensor set mismatch: missing={sorted(set(expected) - set(actual))[:8]}, "
            f"extra={sorted(set(actual) - set(expected))[:8]}"
        )
    data_start = MODEL_HEADER.size + len(expected) * PARAMETER_DESCRIPTOR.size
    next_offset = data_start
    for name, descriptor in sorted(actual.items(), key=lambda item: item[1][3]):
        dtype, shape, size, offset = descriptor
        if (dtype, shape, size) != expected[name]:
            raise AssertionError(f"{name}: expected {expected[name]}, got {(dtype, shape, size)}")
        if offset != next_offset:
            raise AssertionError(f"{name}: expected offset {next_offset}, got {offset}")
        next_offset += size
    if next_offset != file_size:
        raise AssertionError(f"Converted data ends at {next_offset}, file size is {file_size}")
    quantized = sum(dtype == BYTE for dtype, _, _, _ in actual.values())
    print(
        f"LING3_CONVERTED_MODEL_OK model={args.model} tensors={len(actual)} "
        f"kai_w4a32={quantized} bytes={file_size}"
    )


if __name__ == "__main__":
    main()
