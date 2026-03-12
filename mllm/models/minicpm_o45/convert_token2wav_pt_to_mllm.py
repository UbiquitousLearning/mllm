#!/usr/bin/env python3
# Copyright (c) MLLM Team.
# Licensed under the MIT License.

"""Lightweight MiniCPM-o-4_5 token2wav converter.

This script merges `flow.pt` + `hift.pt` into one `.mllm` file without
depending on `pymllm`/`tvm_ffi`.
"""

from __future__ import annotations

import argparse
import gc
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:
    import torch
except ImportError as exc:
    raise ImportError("PyTorch is required. Please install torch in your Python env.") from exc


# ----------------------------- MLLM constants ----------------------------- #

MLLM_MODEL_FILE_V1_MAGIC_NUMBER = 20012
MLLM_MODEL_FILE_V2_MAGIC_NUMBER = 0x519A
MLLM_MODEL_FILE_V2_VERSION = 2
MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH = 512
MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH = 256
MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH = 16

MODEL_FILE_V2_DESC_SIZE = 532
MODEL_FILE_V2_PARAM_DESC_SIZE = 352


def _build_torch_type_mapping() -> Dict[torch.dtype, int]:
    mapping = {
        torch.float32: 0,  # kFloat32
        torch.float16: 1,  # kFloat16
        torch.bfloat16: 128,  # kBFloat16
        torch.int8: 16,  # kInt8
        torch.int16: 17,  # kInt16
        torch.int32: 18,  # kInt32
        torch.int64: 132,  # kInt64
        torch.uint8: 129,  # kUInt8
        torch.bool: 129,  # kUInt8
    }
    if hasattr(torch, "uint16"):
        mapping[torch.uint16] = 130  # kUInt16
    return mapping


TORCH_TYPE_MAPPING = _build_torch_type_mapping()


# ----------------------------- Helpers ----------------------------- #


@dataclass
class TensorMeta:
    raw_name: str
    full_name: str
    dtype_id: int
    data_len: int


def _load_pt(path: Path) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if isinstance(obj, dict):
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
        for candidate in ("state_dict", "model", "module"):
            cand = obj.get(candidate)
            if isinstance(cand, dict) and cand and any(torch.is_tensor(v) for v in cand.values()):
                return {k: v for k, v in cand.items() if torch.is_tensor(v)}
        tensor_only = {k: v for k, v in obj.items() if torch.is_tensor(v)}
        if tensor_only:
            return tensor_only

    raise ValueError(f"Unsupported checkpoint layout: {path}")


def _normalized_tensor(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().cpu().contiguous()
    if x.dim() == 0:
        x = x.reshape(1)
    return x


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    x = _normalized_tensor(t)
    return x.view(torch.uint8).numpy().tobytes()


def _tensor_dtype_id(dtype: torch.dtype) -> int:
    if dtype not in TORCH_TYPE_MAPPING:
        raise ValueError(f"Unsupported tensor dtype for .mllm export: {dtype}")
    return TORCH_TYPE_MAPPING[dtype]


def _collect_source_meta(
    ckpt_path: Path,
    out_prefix: str,
    strip_prefix: str,
    preview_limit: int,
) -> List[TensorMeta]:
    state = _load_pt(ckpt_path)
    keys = list(state.keys())
    print(f"[inspect] {ckpt_path}: {len(keys)} tensors")

    metas: List[TensorMeta] = []
    for i, raw_name in enumerate(keys):
        t = state[raw_name]
        out_name = raw_name[len(strip_prefix) :] if (strip_prefix and raw_name.startswith(strip_prefix)) else raw_name
        full_name = f"{out_prefix}{out_name}"
        x = _normalized_tensor(t)
        dtype_id = _tensor_dtype_id(x.dtype)
        data_len = int(x.numel()) * int(x.element_size())
        metas.append(TensorMeta(raw_name=raw_name, full_name=full_name, dtype_id=dtype_id, data_len=data_len))
        if i < max(preview_limit, 0):
            print(f"  - {raw_name}  shape={tuple(x.shape)} dtype={x.dtype}")

    del state
    gc.collect()
    return metas


def _check_duplicate_names(metas: Iterable[TensorMeta]) -> None:
    seen = set()
    for m in metas:
        if m.full_name in seen:
            raise ValueError(f"Duplicated tensor name after rename: {m.full_name}")
        seen.add(m.full_name)


def _stream_source_tensors(
    ckpt_path: Path,
    metas: List[TensorMeta],
) -> Iterable[Tuple[TensorMeta, torch.Tensor]]:
    state = _load_pt(ckpt_path)
    try:
        for m in metas:
            if m.raw_name not in state:
                raise KeyError(f"Tensor missing in checkpoint: {ckpt_path} -> {m.raw_name}")
            t = _normalized_tensor(state[m.raw_name])
            yield m, t
    finally:
        del state
        gc.collect()


# ----------------------------- V1 writer ----------------------------- #


def _write_v1(
    output: Path,
    model_name: str,
    flow_pt: Path,
    flow_metas: List[TensorMeta],
    hift_pt: Path,
    hift_metas: List[TensorMeta],
) -> None:
    del model_name  # v1 header has no model name

    all_metas = flow_metas + hift_metas
    _check_duplicate_names(all_metas)

    desc_size = 0
    for m in all_metas:
        name_bytes = m.full_name.encode("utf-8")
        desc_size += 4 + len(name_bytes) + 8 + 8 + 4

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(struct.pack("<IQ", MLLM_MODEL_FILE_V1_MAGIC_NUMBER, desc_size))

        current_data_offset = 12 + desc_size
        for m in all_metas:
            name_bytes = m.full_name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<Q", m.data_len))
            f.write(struct.pack("<Q", current_data_offset))
            f.write(struct.pack("<i", m.dtype_id))
            current_data_offset += m.data_len

        written = 0
        for m, t in _stream_source_tensors(flow_pt, flow_metas):
            data = _tensor_to_bytes(t)
            if len(data) != m.data_len:
                raise ValueError(f"Tensor byte size changed for {m.full_name}: {len(data)} != {m.data_len}")
            f.write(data)
            written += 1

        for m, t in _stream_source_tensors(hift_pt, hift_metas):
            data = _tensor_to_bytes(t)
            if len(data) != m.data_len:
                raise ValueError(f"Tensor byte size changed for {m.full_name}: {len(data)} != {m.data_len}")
            f.write(data)
            written += 1

    print(f"[done:v1] wrote {written} tensors -> {output}")


# ----------------------------- V2 writer ----------------------------- #


def _pack_v2_file_desc(model_name: str, num_params: int) -> bytes:
    name_bytes = model_name.encode("utf-8")
    name_bytes = name_bytes.ljust(MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH, b"\0")[:MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH]
    return struct.pack(
        f"<II{MLLM_MODEL_FILE_V2_MODEL_NAME_LENGTH}sIQ",
        MLLM_MODEL_FILE_V2_MAGIC_NUMBER,
        MLLM_MODEL_FILE_V2_VERSION,
        name_bytes,
        num_params,
        MODEL_FILE_V2_DESC_SIZE,
    )


def _pack_v2_param_desc(
    param_id: int,
    param_type: int,
    param_size: int,
    param_offset: int,
    shape: Tuple[int, ...],
    name: str,
) -> bytes:
    if len(shape) > MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH:
        raise ValueError(f"Tensor rank > {MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH} is not supported: {name}")

    shape_padded = list(shape) + [0] * (MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH - len(shape))
    name_bytes = name.encode("utf-8")
    name_bytes = name_bytes.ljust(MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH, b"\0")[:MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH]
    return struct.pack(
        f"<IIQQQ{MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH}i{MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH}s",
        param_id,
        param_type,
        param_size,
        param_offset,
        len(shape),
        *shape_padded,
        name_bytes,
    )


class _V2StreamingWriter:
    def __init__(self, output: Path, model_name: str, max_params: int):
        self.output = output
        self.model_name = model_name
        self.max_params = max_params
        self.num_params = 0
        self.f = open(output, "wb")

        reserved_bytes = MODEL_FILE_V2_DESC_SIZE + max_params * MODEL_FILE_V2_PARAM_DESC_SIZE
        self.f.write(b"\x00" * reserved_bytes)
        self.f.flush()

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if self.num_params >= self.max_params:
            raise ValueError(f"Descriptor buffer exceeded: {self.num_params} >= {self.max_params}")

        dtype_id = _tensor_dtype_id(tensor.dtype)
        shape = tuple(int(v) for v in tensor.shape)
        data = _tensor_to_bytes(tensor)
        data_offset = self.f.tell()
        data_len = len(data)

        self.f.write(data)

        desc_off = MODEL_FILE_V2_DESC_SIZE + self.num_params * MODEL_FILE_V2_PARAM_DESC_SIZE
        self.f.seek(desc_off, os.SEEK_SET)
        self.f.write(
            _pack_v2_param_desc(
                param_id=self.num_params,
                param_type=dtype_id,
                param_size=data_len,
                param_offset=data_offset,
                shape=shape,
                name=name,
            )
        )
        self.f.seek(0, os.SEEK_END)
        self.num_params += 1

    def finalize(self) -> None:
        self.f.seek(0, os.SEEK_SET)
        self.f.write(_pack_v2_file_desc(self.model_name, self.num_params))
        self.f.flush()

    def close(self) -> None:
        if not self.f.closed:
            self.f.close()


def _write_v2(
    output: Path,
    model_name: str,
    flow_pt: Path,
    flow_metas: List[TensorMeta],
    hift_pt: Path,
    hift_metas: List[TensorMeta],
    max_param_desc: int,
) -> None:
    all_metas = flow_metas + hift_metas
    _check_duplicate_names(all_metas)

    if max_param_desc <= 0:
        max_param_desc = len(all_metas)
    if max_param_desc < len(all_metas):
        raise ValueError(f"--max-param-desc ({max_param_desc}) < total tensors ({len(all_metas)})")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = _V2StreamingWriter(output=output, model_name=model_name, max_params=max_param_desc)
    written = 0
    try:
        for m, t in _stream_source_tensors(flow_pt, flow_metas):
            writer.write_tensor(m.full_name, t)
            written += 1
        for m, t in _stream_source_tensors(hift_pt, hift_metas):
            writer.write_tensor(m.full_name, t)
            written += 1
        writer.finalize()
    finally:
        writer.close()

    print(f"[done:v2] wrote {written} tensors -> {output}")


# ----------------------------- Main ----------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MiniCPM-o-4_5 token2wav flow.pt + hift.pt into one .mllm file."
    )
    parser.add_argument(
        "--flow-pt",
        default="mllm/models/minicpm_o45/python_src_code/assets/token2wav/flow.pt",
        help="Path to flow.pt",
    )
    parser.add_argument(
        "--hift-pt",
        default="mllm/models/minicpm_o45/python_src_code/assets/token2wav/hift.pt",
        help="Path to hift.pt",
    )
    parser.add_argument("--output", required=True, help="Output .mllm path")
    parser.add_argument("--model-name", default="minicpm_o45_token2wav", help="Model name (used in v2 header)")
    parser.add_argument("--format", choices=["v1", "v2"], default="v1", help="Output model format")
    parser.add_argument("--flow-prefix", default="token2wav.flow_model.", help="Prefix for flow tensor names")
    parser.add_argument("--hift-prefix", default="token2wav.hift_model.", help="Prefix for hift tensor names")
    parser.add_argument(
        "--strip-hift-prefix",
        default="generator.",
        help="Strip this prefix from hift tensor names before adding --hift-prefix",
    )
    parser.add_argument(
        "--max-param-desc",
        type=int,
        default=0,
        help="Only for v2: max descriptor buffer size, 0 means auto",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Only inspect checkpoints and quit")
    parser.add_argument("--preview-limit", type=int, default=8, help="How many tensors to print per checkpoint")
    args = parser.parse_args()

    flow_pt = Path(args.flow_pt).expanduser().resolve()
    hift_pt = Path(args.hift_pt).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    flow_metas = _collect_source_meta(flow_pt, args.flow_prefix, "", args.preview_limit)
    hift_metas = _collect_source_meta(hift_pt, args.hift_prefix, args.strip_hift_prefix, args.preview_limit)

    total = len(flow_metas) + len(hift_metas)
    print(f"[count] flow={len(flow_metas)}, hift={len(hift_metas)}, total={total}")

    if args.inspect_only:
        print("[inspect-only] done")
        return

    if args.format == "v1":
        _write_v1(
            output=output,
            model_name=args.model_name,
            flow_pt=flow_pt,
            flow_metas=flow_metas,
            hift_pt=hift_pt,
            hift_metas=hift_metas,
        )
    else:
        _write_v2(
            output=output,
            model_name=args.model_name,
            flow_pt=flow_pt,
            flow_metas=flow_metas,
            hift_pt=hift_pt,
            hift_metas=hift_metas,
            max_param_desc=args.max_param_desc,
        )


if __name__ == "__main__":
    main()
