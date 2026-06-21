# Copyright (c) MLLM Team.
# Licensed under the MIT License.

from __future__ import annotations
import os
import sys

__all__ = []


def _has_mobile_libs() -> bool:
    parent_dir = os.path.dirname(os.path.realpath(__file__))

    # Platform-specific library names
    if sys.platform.startswith("win32"):
        lib_name = "MllmFFIExtension.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = "MllmFFIExtension.dylib"
    else:
        lib_name = "MllmFFIExtension.so"

    lib_path = os.path.join(parent_dir, "lib", lib_name)
    return os.path.exists(lib_path)


def is_mobile_available() -> bool:
    return _has_mobile_libs()


if _has_mobile_libs():
    from . import mobile

    __all__.append("mobile")
