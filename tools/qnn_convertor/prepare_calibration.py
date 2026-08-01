#!/usr/bin/env python3
"""Compress a JSONL calibration corpus for the mllm QNN profiler."""

import argparse
from pathlib import Path

import zstandard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output .jsonl.zst path")
    args = parser.parse_args()

    output = args.output or Path(f"{args.input}.zst")
    output.parent.mkdir(parents=True, exist_ok=True)
    compressor = zstandard.ZstdCompressor(level=3)
    output.write_bytes(compressor.compress(args.input.read_bytes()))
    print(output)


if __name__ == "__main__":
    main()
