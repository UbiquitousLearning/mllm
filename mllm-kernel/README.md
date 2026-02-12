# mllm-kernel

High-performance JIT kernels for mllm with Highway SIMD support.

## Features

- **CPU Kernels**: SIMD-accelerated kernels using [Highway](https://github.com/google/highway)
  - Portable across x86 (SSE/AVX/AVX-512), ARM (NEON/SVE), and RISC-V (RVV)
  - Runtime dispatch to best available SIMD instructions
- **CUDA Kernels**: GPU-accelerated kernels (coming soon)
- **Ascend Kernels**: NPU-accelerated kernels (coming soon)

## Installation

### CPU Kernels (with Highway)

```bash
# Install from pyproject-cpu.toml
pip install . --config-settings=cmake.args="-DMLLM_KERNEL_BUILD_CPU=ON"

# Or using the specific config file
pip install . -C pyproject.toml=pyproject-cpu.toml
```

### Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/mllm/mllm-kernel.git
cd mllm-kernel
pip install -e . --config-settings=cmake.args="-DMLLM_KERNEL_BUILD_CPU=ON"
```

## Usage

### Add Constant (CPU with Highway SIMD)

```python
import torch
from mllm_kernel.cpu.jit import add_constant, add_constant_runtime

# Create input tensor
x = torch.randn(1024, dtype=torch.float32)

# Use compile-time constant (faster, limited to predefined values)
y = add_constant(x, 16)  # y = x + 16

# Use runtime constant (flexible, any float value)
y = add_constant_runtime(x, 3.14159)  # y = x + 3.14159
```

### Custom Kernels

You can create your own Highway-accelerated kernels:

```python
from mllm_kernel.jit_utils import load_cpu_jit, make_cpp_args, cache_once

@cache_once
def _jit_my_kernel_module(param: int):
    args = make_cpp_args(param)
    return load_cpu_jit(
        "my_kernel", *args,
        cpp_files=["my_kernel.cpp"],
        cpp_wrappers=[("my_kernel", f"my_namespace::my_kernel<{args}>")],
    )

def my_kernel(src: torch.Tensor, param: int) -> torch.Tensor:
    dst = torch.empty_like(src)
    module = _jit_my_kernel_module(param)
    module.my_kernel(dst, src)
    return dst
```

### Generate Recommended `.clangd` Config

Use the helper command below to generate a recommended `.clangd` file for this
repository:

```bash
cd ./mllm-kernel
python -m mllm_kernel show-clangd-recommend-config
```

What this command does:

- Adds include paths required by `tvm_ffi`, DLPack, and `mllm-kernel`
- Detects GPU compute capability from `nvidia-smi` and sets `--cuda-gpu-arch`
- Tries to detect CUDA Toolkit path and appends `--cuda-path=...` when found
- If `.clangd` already exists, it will not overwrite it and prints suggested content

Tip: if CUDA is installed in a non-standard location, set `CUDA_HOME` or
`CUDA_PATH` before running the command.

## Project Structure

```
mllm-kernel/
├── cmake/
│   ├── CPM.cmake              # CMake Package Manager
│   └── MllmKernelConfig.cmake.in
├── mllm_kernel/
│   ├── __init__.py
│   ├── cpu/
│   │   ├── __init__.py
│   │   ├── csrc/              # C++ kernel source files
│   │   │   └── add_constant.cpp
│   │   ├── include/           # Header files
│   │   │   └── mllm_kernel/
│   │   │       ├── common.h
│   │   │       └── simd_ops.h
│   │   └── jit/               # Python wrappers
│   │       ├── __init__.py
│   │       └── add_constant.py
│   ├── cuda/                  # CUDA kernels (future)
│   ├── ascend/                # Ascend kernels (future)
│   └── jit_utils/
│       ├── __init__.py
│       ├── cache.py
│       └── compile.py         # JIT compilation utilities
├── CMakeLists.txt
├── pyproject-cpu.toml         # CPU package config
├── pyproject-cuda.toml        # CUDA package config
└── README.md
```

## Writing Highway Kernels

### Basic Pattern

```cpp
#include "mllm_kernel/common.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace mllm_kernel {
namespace cpu {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

template <typename T>
HWY_ATTR void MyKernel(T* dst, const T* src, size_t n) {
    const hn::ScalableTag<T> d;
    const size_t lanes = hn::Lanes(d);
    
    size_t i = 0;
    // SIMD loop
    for (; i + lanes <= n; i += lanes) {
        auto v = hn::LoadU(d, src + i);
        // ... process v ...
        hn::StoreU(v, d, dst + i);
    }
    
    // Scalar tail
    for (; i < n; ++i) {
        dst[i] = /* scalar operation */;
    }
}

}  // namespace HWY_NAMESPACE
}  // namespace cpu
}  // namespace mllm_kernel
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace mllm_kernel {
namespace cpu {

HWY_EXPORT(MyKernel<float>);

struct my_kernel {
    void operator()(tvm_ffi::NDArray dst, tvm_ffi::NDArray src) const {
        HWY_DYNAMIC_DISPATCH(MyKernel<float>)(
            GetDataPtr<float>(dst),
            GetConstDataPtr<float>(src),
            GetNumElements(src)
        );
    }
};

}  // namespace cpu
}  // namespace mllm_kernel
#endif
```

## Dependencies

- Python >= 3.10
- PyTorch
- apache-tvm-ffi >= 0.1.0b4
- CMake >= 3.21
- C++20 compatible compiler

Highway is automatically downloaded and built during installation via CPM.

## License

Apache-2.0
