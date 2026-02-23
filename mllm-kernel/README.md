# mllm-kernel

JIT-compiled CPU and CUDA kernel helpers for `mllm`, built on top of `tvm_ffi`.

## Current Status

- CPU JIT path is available via `mllm_kernel.cpu.jit`.
- CUDA JIT path is available via `mllm_kernel.cuda.jit`.
- Ascend build flag exists in CMake, but there is no Python Ascend package in this repository yet.
- The `add_constant` kernels are currently scaffold/demo code and should be treated as examples, not production-validated ops.

## Requirements

- Python >= 3.10
- CMake >= 3.21
- C++20 compiler
- PyTorch
- `apache-tvm-ffi`
- `torch-c-dlpack-ext`
- CUDA toolkit and driver (only if you use CUDA JIT kernels)

## Installation

Install from source:

```bash
cd mllm-kernel
pip install .
```

Editable install for development:

```bash
cd mllm-kernel
pip install -e ".[dev]"
```

### Build Options

Default build options are defined in `pyproject.toml`:

- `MLLM_KERNEL_BUILD_CPU=ON`
- `MLLM_KERNEL_BUILD_CUDA=ON`
- `MLLM_KERNEL_BUILD_ASCEND=OFF`

You can override CMake options at install time, for example:

```bash
pip install . --config-settings=cmake.args="-DMLLM_KERNEL_BUILD_CPU=ON;-DMLLM_KERNEL_BUILD_CUDA=OFF"
```

## Quick Usage

### CPU JIT (`float32`)

```python
import torch
from mllm_kernel.cpu.jit import add_constant, add_constant_runtime

x = torch.randn(1024, dtype=torch.float32)

# Compile-time constant (supported: 1, 2, 4, 8, 16)
y1 = add_constant(x, 16)

# Runtime float constant
y2 = add_constant_runtime(x, 3.14159)
```

### CUDA JIT (`int32`, CUDA device tensor)

```python
import torch
from mllm_kernel.cuda.jit import add_constant

x = torch.arange(1024, dtype=torch.int32, device="cuda")
y = add_constant(x, 8)
```

## Writing Custom JIT Kernels

Use the helpers in `mllm_kernel.jit_utils`:

- `load_cpu_jit`
- `load_cuda_jit`
- `make_cpp_args`
- `cache_once`

Example pattern:

```python
import torch
from mllm_kernel.jit_utils import cache_once, load_cpu_jit, make_cpp_args

@cache_once
def _jit_my_kernel_module(param: int):
    args = make_cpp_args(param)
    return load_cpu_jit(
        "my_kernel",
        *args,
        cpp_files=["my_kernel.cpp"],
        cpp_wrappers=[("my_kernel", f"my_namespace::my_kernel<{args}>")],
    )

def my_kernel(src: torch.Tensor, param: int) -> torch.Tensor:
    dst = torch.empty_like(src)
    module = _jit_my_kernel_module(param)
    module.my_kernel(dst, src)
    return dst
```

## Generate `.clangd` Config

From repository root:

```bash
python -m mllm_kernel show-clangd-recommend-config
```

This helper:

- Adds include paths for `tvm_ffi`, DLPack, and this package.
- Detects compute capability with `nvidia-smi` and sets `--cuda-gpu-arch`.
- Tries to auto-detect CUDA toolkit path and appends `--cuda-path=...`.
- Refuses to overwrite an existing `.clangd` file.

If CUDA is installed in a non-standard location, set `CUDA_HOME` or `CUDA_PATH` first.

## Show Runtime Environment

From repository root:

```bash
python -m mllm_kernel show-env
```

This helper prints runtime details, with CUDA information as the primary focus:

- CUDA toolkit and driver hints (`CUDA_HOME`/`CUDA_PATH`, `nvcc`, `nvidia-smi`)
- CUDA versions from `nvcc`, `nvidia-smi`, and PyTorch build/runtime
- GPU list and compute capability (when CUDA is available in PyTorch)
- OS, Python, and CPU model/core count

## Show JIT Registration and Cache Status

From repository root:

```bash
python -m mllm_kernel show-config
```

This helper:

- Prints the same environment summary as `show-env`.
- Lists registered JIT kernels by device.
- Shows whether each kernel has cached `.so` artifacts.
- Uses `~/.cache/mllm_kernel` as the default cache root.

Typical first-run behavior:

- First invocation of a kernel may take longer because it triggers compilation.
- Later invocations reuse cached artifacts and are much faster.

## Project Layout

```text
mllm-kernel/
├── CMakeLists.txt
├── pyproject.toml
├── include/mllm_kernel/
│   ├── source_location.hpp
│   ├── tensor.hpp
│   ├── utils.hpp
│   └── utils.cuh
├── mllm_kernel/
│   ├── __main__.py
│   ├── cpu/
│   │   ├── csrc/add_constant.cpp
│   │   └── jit/add_constant.py
│   ├── cuda/
│   │   ├── csrc/add_constant.cuh
│   │   └── jit/add_constant.py
│   └── jit_utils/compile.py
└── cmake/
```
