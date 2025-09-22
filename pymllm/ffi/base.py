# Copyright (c) MLLM Team.
# Licensed under the MIT License.

import tvm_ffi
import os
import sys


def _load_lib():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.dirname(file_dir)

    # Platform-specific library names
    if sys.platform.startswith("win32"):
        lib_name = "MllmFFIExtension.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = "MllmFFIExtension.dylib"
    else:
        lib_name = "MllmFFIExtension.so"

    lib_path = os.path.join(parent_dir, "lib", lib_name)
    return tvm_ffi.load_module(lib_path)


_LIB = _load_lib()
