from __future__ import annotations

import torch
import pathlib
import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    Literal,
)

if TYPE_CHECKING:
    from tvm_ffi import Module


F = TypeVar("F", bound=Callable[..., Any])


def cache_once(fn: F) -> F:
    """
    NOTE: `functools.lru_cache` is not compatible with `torch.compile`
    So we manually implement a simple cache_once decorator to replace it.
    """
    result_map = {}

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items(), key=lambda x: x[0])))
        if key not in result_map:
            result_map[key] = fn(*args, **kwargs)
        return result_map[key]

    return wrapper  # type: ignore


def _make_tvm_ffi_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


@cache_once
def _resolve_kernel_path() -> pathlib.Path:
    cur_dir = pathlib.Path(__file__).parent.resolve()

    # first, try this directory structure
    def _environment_install():
        candidate = cur_dir.resolve()
        if (candidate / "cpu").exists() and (candidate / "cuda").exists():
            return candidate
        return None

    path = _environment_install()
    if path is None:
        raise RuntimeError("Cannot find mllm-kernel/jit path")
    return path


@cache_once
def _resolve_cpu_simd_features_to_cxx_flags() -> list[str]:
    pass


class _MLLMKernelTemplateArgListGenerator(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


MLLM_KERNEL_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool, torch.dtype]
MLLM_KERNEL_TEMPLATE_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float: "fp32_t",
    torch.float16: "fp16_t",
    torch.bfloat16: "bfp16_t",
    torch.int8: "int8_t",
    torch.int16: "int16_t",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.uint8: "uint8_t",
    torch.uint16: "uint16_t",
    torch.uint32: "uint32_t",
    torch.uint64: "uint64_t",
    torch.bool: "bool_t",
}


def make_cpp_args(
    *args: MLLM_KERNEL_TEMPLATE_TYPE,
) -> _MLLMKernelTemplateArgListGenerator:
    def _convert(arg: MLLM_KERNEL_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return MLLM_KERNEL_TEMPLATE_DTYPE_MAP[arg]
        raise TypeError(
            f"Unsupported argument type for mllm kernel template: {type(arg)}"
        )

    return _MLLMKernelTemplateArgListGenerator(_convert(arg) for arg in args)


MLLM_KERNEL_TOP_PATH = _resolve_kernel_path()
MLLM_KERNEL_CPU_PATH = MLLM_KERNEL_TOP_PATH / "cpu"
MLLM_KERNEL_CUDA_PATH = MLLM_KERNEL_TOP_PATH / "cuda"
MLLM_KERNEL_ASCEND_PATH = MLLM_KERNEL_TOP_PATH / "ascend"

MLLM_KERNEL_CPU_CSRC_DIR = MLLM_KERNEL_CPU_PATH / "csrc"
MLLM_KERNEL_CPU_INCLUDE_DIR = MLLM_KERNEL_CPU_PATH / "include"

MLLM_KERNEL_CUDA_CSRC_DIR = MLLM_KERNEL_CUDA_PATH / "csrc"
MLLM_KERNEL_CUDA_INCLUDE_DIR = MLLM_KERNEL_CUDA_PATH / "include"

MLLM_KERNEL_ASCEND_CSRC_DIR = MLLM_KERNEL_ASCEND_PATH / "csrc"
MLLM_KERNEL_ASCEND_INCLUDE_DIR = MLLM_KERNEL_ASCEND_PATH / "include"

MLLM_KERNEL_DEFAULT_CXX_FLAGS = ["-std=c++20", "-O3"]
MLLM_KERNEL_DEFAULT_CUDA_C_FLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]

MLLM_KERNEL_DEFAULT_LDFLAGS = []


def _tvm_ffi_cpp_load_inline(
    *args: str,
    device: Literal["cpu", "cuda", "ascend"] = "cpu",
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cxx_flags: List[str] | None = None,
    extra_cuda_cxx_flags: List[str] | None = None,
    extra_ld_flags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load_inline

    if device == "ascend":
        raise NotImplementedError("Ascend is not supported yet")

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cxx_flags = extra_cxx_flags or []
    extra_cuda_cxx_flags = extra_cuda_cxx_flags or []
    extra_ld_flags = extra_ld_flags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(MLLM_KERNEL_CPU_CSRC_DIR / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_tvm_ffi_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(MLLM_KERNEL_CUDA_CSRC_DIR / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_tvm_ffi_wrapper(tup) for tup in cuda_wrappers]

    MLLM_KERNEL_DEFAULT_INCLUDE_DIRS = (
        MLLM_KERNEL_CPU_INCLUDE_DIR if device == "cpu" else MLLM_KERNEL_CUDA_INCLUDE_DIR
    )

    return load_inline(
        "mllm_jit_kernel_" + device + "_" + "_".join(str(arg) for arg in args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=MLLM_KERNEL_DEFAULT_CXX_FLAGS + extra_cxx_flags,
        extra_cuda_cflags=MLLM_KERNEL_DEFAULT_CUDA_C_FLAGS + extra_cuda_cxx_flags,
        extra_ldflags=MLLM_KERNEL_DEFAULT_LDFLAGS + extra_ld_flags,
        extra_include_paths=MLLM_KERNEL_DEFAULT_INCLUDE_DIRS + extra_include_paths,
        build_directory=build_directory,
    )
