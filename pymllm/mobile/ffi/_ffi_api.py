# Copyright (c) MLLM Team.
# Licensed under the MIT License.

# _ffi_api.py
import tvm_ffi
from .base import _LIB

# Register all global functions prefixed with 'mllm.'
# This makes functions registered via TVM_FFI_STATIC_INIT_BLOCK available
tvm_ffi.init_ffi_api("mllm", __name__)
