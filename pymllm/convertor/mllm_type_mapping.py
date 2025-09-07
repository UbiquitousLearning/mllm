# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import torch
import numpy as np

MLLM_TYPE_MAPPING = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 128,
}
