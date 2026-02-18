# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import json
import tvm_ffi

from ..ffi import Session


def start_service(workers: int) -> None:
    tvm_ffi.get_global_func("mllm.service.startService")(workers)


def stop_service() -> None:
    tvm_ffi.get_global_func("mllm.service.stopService")()


def send_request(request: str) -> str:
    return tvm_ffi.get_global_func("mllm.service.sendRequest")(request)


def get_response(id: str):
    stop = False
    while not stop:
        ret_json = tvm_ffi.get_global_func("mllm.service.getResponse")(id)
        stop = json.loads(ret_json)["choices"][0]["finish_reason"] == "stop"
        yield ret_json


def insert_session(id: str, s: Session) -> None:
    tvm_ffi.get_global_func("mllm.service.insertSession")(id, s)
