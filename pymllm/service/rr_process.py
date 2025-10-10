# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import tvm_ffi
from ..ffi import Session


def start_service() -> None:
    tvm_ffi.get_global_func("mllm.service.startService")()


def stop_service() -> None:
    tvm_ffi.get_global_func("mllm.service.stopService")()


def send_request(request: str) -> str:
    return tvm_ffi.get_global_func("mllm.service.sendRequest")(request)


def get_response() -> str:
    return tvm_ffi.get_global_func("mllm.service.getResponse")()


def insert_session(s: Session) -> None:
    tvm_ffi.get_global_func("mllm.service.insertSession")(s)
