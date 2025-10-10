# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import tvm_ffi
from ..ffi import Session


def session_qwen3(fp: str) -> Session:
    return tvm_ffi.get_global_func("mllm.service.session.qwen3")(fp)


MODEL_HUB_LOOKUP_TABLE = {
    "mllmTeam/Qwen3-0.6B-w4a32kai": session_qwen3,
}
