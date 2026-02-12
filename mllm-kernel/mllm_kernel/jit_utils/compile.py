from __future__ import annotations

import torch
import pathlib
import functools
import os
import sys
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
    """Resolve the path to the mllm_kernel package directory."""
    cur_dir = pathlib.Path(__file__).parent.parent.resolve()

    # Check if running from source tree
    if (cur_dir / "cpu").exists():
        return cur_dir

    # Check if installed as a package
    if (cur_dir / "include").exists():
        return cur_dir

    raise RuntimeError(f"Cannot find mllm-kernel path. Searched in: {cur_dir}")


@cache_once
def _resolve_highway_include_path() -> pathlib.Path | None:
    """
    Resolve the path to Highway headers.

    Search order:
    1. Installed package include directory
    2. System include paths
    3. CPM cache directory
    """
    # 1. Check installed package directory
    kernel_path = _resolve_kernel_path()
    pkg_include = kernel_path / "include"
    if (pkg_include / "hwy").exists():
        return pkg_include

    # Also check parent include (when installed via pip)
    parent_include = kernel_path.parent / "include"
    if (parent_include / "hwy").exists():
        return parent_include

    # 2. Check site-packages include
    for site_pkg in sys.path:
        site_include = pathlib.Path(site_pkg) / "include"
        if (site_include / "hwy").exists():
            return site_include

    # 3. Check CPM cache (for development)
    cpm_cache = pathlib.Path.home() / ".cache" / "CPM"
    if cpm_cache.exists():
        for hwy_dir in cpm_cache.glob("highway/*/hwy"):
            return hwy_dir.parent

    # 4. Check mllm build directory (for development with parent project)
    mllm_root = kernel_path.parent.parent
    mllm_build = mllm_root / "build" / "_deps" / "highway-src"
    if (mllm_build / "hwy").exists():
        return mllm_build

    return None


@cache_once
def _get_cpu_arch_flags() -> List[str]:
    """
    Get CPU architecture-specific compiler flags.

    Returns appropriate SIMD flags based on the current platform.
    """

    # TODO

    return []


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
    """
    Convert Python arguments to C++ template argument strings.

    Example:
        make_cpp_args(16, torch.float16) -> ["16", "fp16_t"]
    """

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


# Path constants
MLLM_KERNEL_TOP_PATH = _resolve_kernel_path()
MLLM_KERNEL_INCLUDE_DIR = MLLM_KERNEL_TOP_PATH.parent / "include"
MLLM_KERNEL_CPU_PATH = MLLM_KERNEL_TOP_PATH / "cpu"
MLLM_KERNEL_CUDA_PATH = MLLM_KERNEL_TOP_PATH / "cuda"
MLLM_KERNEL_ASCEND_PATH = MLLM_KERNEL_TOP_PATH / "ascend"

MLLM_KERNEL_CPU_CSRC_DIR = MLLM_KERNEL_CPU_PATH / "csrc"
MLLM_KERNEL_CPU_INCLUDE_DIR = MLLM_KERNEL_CPU_PATH / "include"

MLLM_KERNEL_CUDA_CSRC_DIR = MLLM_KERNEL_CUDA_PATH / "csrc"
MLLM_KERNEL_CUDA_INCLUDE_DIR = MLLM_KERNEL_CUDA_PATH / "include"

MLLM_KERNEL_ASCEND_CSRC_DIR = MLLM_KERNEL_ASCEND_PATH / "csrc"
MLLM_KERNEL_ASCEND_INCLUDE_DIR = MLLM_KERNEL_ASCEND_PATH / "include"

# Compiler flags
MLLM_KERNEL_DEFAULT_CXX_FLAGS = ["-std=c++20", "-O3", "-fPIC"]
MLLM_KERNEL_DEFAULT_CUDA_C_FLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
MLLM_KERNEL_DEFAULT_LDFLAGS = []


def load_cpu_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    extra_cxx_flags: List[str] | None = None,
    extra_ld_flags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    use_highway: bool = True,
    build_directory: str | None = None,
) -> Module:
    """
    Load a CPU JIT kernel module using Highway SIMD.

    Args:
        args: Unique identifiers for this kernel module
        cpp_files: List of C++ source files (relative to cpu/csrc/)
        cpp_wrappers: List of (export_name, kernel_name) tuples
        extra_cxx_flags: Additional C++ compiler flags
        extra_ld_flags: Additional linker flags
        extra_include_paths: Additional include paths
        use_highway: Whether to include Highway SIMD library (default: True)
        build_directory: Custom build directory for JIT cache

    Returns:
        Compiled JIT module

    Example:
        module = load_cpu_jit(
            "add_constant", "16",
            cpp_files=["add_constant.cpp"],
            cpp_wrappers=[("add_constant", "add_constant<16>")],
        )
        module.add_constant(output, input)
    """
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cpp_wrappers = cpp_wrappers or []
    extra_cxx_flags = extra_cxx_flags or []
    extra_ld_flags = extra_ld_flags or []
    extra_include_paths = extra_include_paths or []

    # Build include paths
    include_paths = [str(MLLM_KERNEL_CPU_INCLUDE_DIR), str(MLLM_KERNEL_INCLUDE_DIR)]

    # Add Highway include path if requested
    if use_highway:
        hwy_include = _resolve_highway_include_path()
        if hwy_include is not None:
            include_paths.append(str(hwy_include))
        else:
            raise RuntimeError(
                "Highway SIMD library not found. "
                "Please install mllm-kernel-cpu with: pip install mllm-kernel-cpu"
            )

    include_paths.extend(extra_include_paths)

    # Build compiler flags
    cxx_flags = MLLM_KERNEL_DEFAULT_CXX_FLAGS.copy()
    cxx_flags.extend(_get_cpu_arch_flags())
    cxx_flags.extend(extra_cxx_flags)

    # Build C++ sources (include cpp files)
    cpp_paths = [(MLLM_KERNEL_CPU_CSRC_DIR / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_tvm_ffi_wrapper(tup) for tup in cpp_wrappers]

    return load_inline(
        "mllm_jit_kernel_cpu_" + "_".join(str(arg) for arg in args),
        cpp_sources=cpp_sources,
        cuda_sources=[],
        extra_cflags=cxx_flags,
        extra_cuda_cflags=[],
        extra_ldflags=MLLM_KERNEL_DEFAULT_LDFLAGS + extra_ld_flags,
        extra_include_paths=include_paths,
        build_directory=build_directory,
    )


def load_cuda_jit(
    *args: str,
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
    """
    Load a CUDA JIT kernel module.

    Args:
        args: Unique identifiers for this kernel module
        cpp_files: List of C++ source files (relative to cuda/csrc/)
        cuda_files: List of CUDA source files (relative to cuda/csrc/)
        cpp_wrappers: List of (export_name, kernel_name) tuples for C++ kernels
        cuda_wrappers: List of (export_name, kernel_name) tuples for CUDA kernels
        extra_cxx_flags: Additional C++ compiler flags
        extra_cuda_cxx_flags: Additional CUDA compiler flags
        extra_ld_flags: Additional linker flags
        extra_include_paths: Additional include paths
        build_directory: Custom build directory for JIT cache

    Returns:
        Compiled JIT module
    """
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cxx_flags = extra_cxx_flags or []
    extra_cuda_cxx_flags = extra_cuda_cxx_flags or []
    extra_ld_flags = extra_ld_flags or []
    extra_include_paths = extra_include_paths or []

    # Build include paths
    include_paths = [str(MLLM_KERNEL_CUDA_INCLUDE_DIR), str(MLLM_KERNEL_INCLUDE_DIR)]
    include_paths.extend(extra_include_paths)

    # Build C++ sources
    cpp_paths = [(MLLM_KERNEL_CUDA_CSRC_DIR / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_tvm_ffi_wrapper(tup) for tup in cpp_wrappers]

    # Build CUDA sources
    cuda_paths = [(MLLM_KERNEL_CUDA_CSRC_DIR / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_tvm_ffi_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        "mllm_jit_kernel_cuda_" + "_".join(str(arg) for arg in args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=MLLM_KERNEL_DEFAULT_CXX_FLAGS + extra_cxx_flags,
        extra_cuda_cflags=MLLM_KERNEL_DEFAULT_CUDA_C_FLAGS + extra_cuda_cxx_flags,
        extra_ldflags=MLLM_KERNEL_DEFAULT_LDFLAGS + extra_ld_flags,
        extra_include_paths=include_paths,
        build_directory=build_directory,
    )
