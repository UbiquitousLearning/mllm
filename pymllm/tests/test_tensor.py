# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import pymllm as torch


def test_empty_tensor_create() -> bool:
    a = torch.empty([1024], torch.float32, "cpu")
    return a.shape() == [1024]


def test_is_torch_available() -> bool:
    return torch.is_torch_available()
