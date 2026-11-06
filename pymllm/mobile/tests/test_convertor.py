# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import json
import struct

import pytest
import torch

from pymllm.mobile.convertor import load_model
from pymllm.mobile.convertor.model_file_v2 import (
    ModelFileV2Descriptor,
    ModelFileV2ParamsDescriptor,
    _pack_descriptor,
)


def test_load_model_filters_torch_checkpoint_by_prefix(tmp_path):
    checkpoint_path = tmp_path / "model.bin"
    torch.save(
        {
            "model.layers.0.weight": torch.ones(1),
            "visual.blocks.0.weight": torch.zeros(1),
        },
        checkpoint_path,
    )

    state_dict = load_model(checkpoint_path, include_prefixes=("model.",))

    assert list(state_dict) == ["model.layers.0.weight"]


def test_model_file_v2_validates_utf8_name_capacity():
    with pytest.raises(ValueError, match="fewer than 512 UTF-8 bytes"):
        ModelFileV2Descriptor(model_name="模" * 171, num_params=0)


def test_model_file_v2_rejects_shape_above_descriptor_rank():
    with pytest.raises(ValueError, match="rank exceeds"):
        ModelFileV2ParamsDescriptor(
            param_id=0,
            param_type=0,
            param_size=1,
            param_offset=0,
            shape=[1] * 17,
            name="weight",
        )


def test_model_file_v2_preserves_parameter_offsets_above_uint32():
    parameter_offset = (1 << 32) + 12345
    descriptor = ModelFileV2ParamsDescriptor(
        param_id=7,
        param_type=134,
        param_size=4096,
        param_offset=parameter_offset,
        shape=[4096],
        name="weight",
    )

    packed = _pack_descriptor(descriptor)
    _, _, _, unpacked_offset, _ = struct.unpack_from("<IIQQQ", packed)

    assert len(packed) == ModelFileV2ParamsDescriptor.SIZE
    assert unpacked_offset == parameter_offset


def _write_indexed_model(model_dir, shard_name):
    from safetensors.torch import save_file

    model_dir.mkdir(parents=True)
    save_file({"probe.weight": torch.full((1,), 23.0)}, model_dir / "shard.safetensors")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"probe.weight": shard_name}})
    )


def test_load_model_reads_indexed_shard_inside_model_dir(tmp_path):
    _write_indexed_model(tmp_path / "model", "shard.safetensors")

    state_dict = load_model(tmp_path / "model")

    assert state_dict["probe.weight"].item() == 23.0


@pytest.mark.parametrize("escape", ["relative", "absolute"])
def test_load_model_rejects_index_shard_outside_model_dir(tmp_path, escape):
    from safetensors.torch import save_file

    outside = tmp_path / "outside.safetensors"
    save_file({"probe.weight": torch.full((1,), 183.0)}, outside)
    shard_name = "../outside.safetensors" if escape == "relative" else str(outside)
    _write_indexed_model(tmp_path / "model", shard_name)

    with pytest.raises(ValueError, match="outside the model directory"):
        load_model(tmp_path / "model")


def test_load_model_follows_symlinked_shard_inside_model_dir(tmp_path):
    # Hugging Face cache snapshots store shards as symlinks into ../../blobs.
    from safetensors.torch import save_file

    blob = tmp_path / "blobs" / "abc123"
    blob.parent.mkdir()
    save_file({"probe.weight": torch.full((1,), 42.0)}, blob)
    _write_indexed_model(tmp_path / "snapshot", "linked.safetensors")
    (tmp_path / "snapshot" / "linked.safetensors").symlink_to(blob)

    state_dict = load_model(tmp_path / "snapshot")

    assert state_dict["probe.weight"].item() == 42.0
