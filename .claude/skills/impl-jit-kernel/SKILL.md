---
name: impl-jit-kernel
description: Guide for implementing CUDA or CPU JIT kernels in mllm-kernel. Use when the user asks to create, add, or implement a new kernel in mllm-kernel.
---

# Implementing a JIT Kernel in mllm-kernel

## Overview

mllm-kernel uses a JIT (Just-In-Time) compilation system built on `tvm_ffi`. Kernels are written in C++20 (`.cuh` for CUDA, `.cpp` for CPU), validated at runtime via `TensorMatcher`, and exposed to Python through a `@jit` decorator. No pre-compilation is needed -- kernels compile on first call and are cached at `~/.cache/mllm_kernel/`.

## File Layout

For a kernel named `my_kernel`:

```
mllm-kernel/
  mllm_kernel/
    cuda/
      csrc/my_kernel.cuh          # CUDA kernel implementation
      jit/my_kernel.py            # Python JIT wrapper
      jit/__init__.py             # Add export here
    cpu/
      csrc/my_kernel.cpp          # CPU kernel implementation (Highway SIMD)
      include/mllm_kernel/cpu/
        my_kernel.hpp             # CPU SIMD body (NO #pragma once)
      jit/my_kernel.py            # Python JIT wrapper
      jit/__init__.py             # Add export here
  tests/test_my_kernel.py         # Pytest correctness tests
  benchmarks/bench_my_kernel.py   # Profiler benchmark vs PyTorch reference
```

---

## CUDA Kernel Walkthrough

### Step 1: Write the `.cuh` kernel

Create `mllm_kernel/cuda/csrc/my_kernel.cuh`:

```cpp
#pragma once

#include <mllm_kernel/tensor.hpp>   // TensorMatcher, SymbolicSize, SymbolicDevice, SymbolicDType
#include <mllm_kernel/utils.hpp>    // RuntimeCheck, Panic, div_ceil
#include <mllm_kernel/utils.cuh>    // LaunchKernel, fp16_t, bf16_t, PDL helpers

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// 1. Parameter struct (trivially copyable, passed to kernel by value)
// ---------------------------------------------------------------------------
struct MyKernelParams {
  const float* __restrict__ input;
  float*       __restrict__ output;
  int32_t num_elements;
};

// ---------------------------------------------------------------------------
// 2. CUDA kernel
// ---------------------------------------------------------------------------
__global__ void my_kernel(const MyKernelParams params) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= params.num_elements) return;
  params.output[idx] = params.input[idx] * 2.0f;
}

// ---------------------------------------------------------------------------
// 3. Host-side launcher (entry point for TVM FFI binding)
// ---------------------------------------------------------------------------
struct MyKernel {
  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView output) {
    using namespace mllm_kernel::host;

    // --- Validate tensors ---
    SymbolicSize N{"num_elements"};
    SymbolicDevice device;

    (void)TensorMatcher({N})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device)
        .verify(input);

    (void)TensorMatcher({N})
        .with_dtype<float>()
        .with_device(device)
        .verify(output);

    const int64_t n = N.unwrap();
    RuntimeCheck(n > 0, "num_elements must be positive, got ", n);

    // --- Build params ---
    MyKernelParams params{
        .input  = static_cast<const float*>(input.data_ptr()),
        .output = static_cast<float*>(output.data_ptr()),
        .num_elements = static_cast<int32_t>(n),
    };

    // --- Launch ---
    constexpr int kBlock = 256;
    const int grid = static_cast<int>(div_ceil(n, kBlock));
    LaunchKernel(grid, kBlock, device.unwrap())(my_kernel, params);
  }
};

}  // namespace
```

**Key rules:**

- **Always wrap in `namespace {}`** (anonymous namespace).
- **Entry point** is a `static void run(tvm::ffi::TensorView ...)` method.
- **Validate every tensor** with `TensorMatcher` before reading `.data_ptr()`.
- **Never dereference device pointers on host** -- `data_ptr()` returns a GPU pointer.
- **Use `LaunchKernel`** to launch -- it handles stream resolution and error checking.

### Step 2: Write the Python JIT wrapper

Create `mllm_kernel/cuda/jit/my_kernel.py`:

```python
"""JIT wrapper for my_kernel CUDA kernel."""

import torch
from mllm_kernel.jit_utils import jit


@jit(
    args=[],
    device="cuda",
    cuda_files=["my_kernel.cuh"],
    cpp_wrappers=[],
    cuda_wrappers=[("my_kernel", "MyKernel::run")],
    func_name="my_kernel",
)
def _kernel(compiled_module, input: torch.Tensor, output: torch.Tensor) -> None:
    compiled_module.my_kernel(input, output)


def my_kernel(input: torch.Tensor) -> torch.Tensor:
    """Double every element in *input*.

    Parameters
    ----------
    input : torch.Tensor
        1-D float32 tensor on CUDA.

    Returns
    -------
    torch.Tensor
        Same shape and dtype as *input*.
    """
    output = torch.empty_like(input)
    _kernel(input, output)
    return output
```

### Step 3: Export in `__init__.py`

Edit `mllm_kernel/cuda/jit/__init__.py` and add:

```python
from mllm_kernel.cuda.jit.my_kernel import my_kernel
```

### Step 4: Clear JIT cache after editing `.cuh`

Any time you modify the `.cuh` file, delete the cached `.so`:

```bash
rm -rf ~/.cache/mllm_kernel/cuda_my_kernel*
```

The next Python call will trigger recompilation automatically.

---

## Template-Parameterized CUDA Kernels

When the kernel takes compile-time constants (e.g. block size, dtype), use `make_cpp_args`:

```python
from mllm_kernel.jit_utils import jit, make_cpp_args

def _make_kernel(block_size: int, use_pdl: bool):
    cpp_args = make_cpp_args(block_size, use_pdl)  # -> "256, true"

    @jit(
        args=[block_size, use_pdl],
        device="cuda",
        cuda_files=["my_kernel.cuh"],
        cpp_wrappers=[],
        cuda_wrappers=[("my_kernel", f"MyKernel<{cpp_args}>::run")],
        func_name="my_kernel",
    )
    def _kernel(compiled_module, input, output):
        compiled_module.my_kernel(input, output)
    return _kernel
```

`make_cpp_args` converts Python types to C++ literals:
- `int/float` -> string literal
- `bool` -> `"true"` / `"false"`
- `torch.dtype` -> C++ type (`torch.float32` -> `"fp32_t"`, `torch.float16` -> `"fp16_t"`, `torch.bfloat16` -> `"bf16_t"`, `torch.int32` -> `"int32_t"`, etc.)

---

## CPU Kernel Walkthrough

CPU kernels use **Google Highway** for portable SIMD. The key difference: the `.hpp` body is included **multiple times** by Highway's `foreach_target` dispatch, so it must NOT have `#pragma once`.

### Step 1: Write the SIMD body (`.hpp`)

Create `mllm_kernel/cpu/include/mllm_kernel/cpu/my_kernel.hpp`:

```cpp
// NOTE: NO #pragma once -- this file is included multiple times by Highway.

#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace mllm_kernel::cpu {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

template <int Constant>
inline void my_kernel_impl(float* HWY_RESTRICT dst,
                           const float* HWY_RESTRICT src,
                           size_t count) {
  const hn::ScalableTag<float> d;
  const size_t lanes = hn::Lanes(d);
  const auto vc = hn::Set(d, static_cast<float>(Constant));
  size_t i = 0;
  for (; i + lanes <= count; i += lanes) {
    const auto v = hn::Load(d, src + i);
    hn::Store(hn::Add(v, vc), d, dst + i);
  }
  for (; i < count; ++i) {
    dst[i] = src[i] + static_cast<float>(Constant);
  }
}

// Named entry points for HWY_EXPORT
static HWY_NOINLINE HWY_MAYBE_UNUSED void my_kernel_1(float* d, const float* s, size_t n) {
  my_kernel_impl<1>(d, s, n);
}

}  // namespace HWY_NAMESPACE
}  // namespace mllm_kernel::cpu
HWY_AFTER_NAMESPACE();
```

### Step 2: Write the `.cpp` source

Create `mllm_kernel/cpu/csrc/my_kernel.cpp`:

```cpp
#include <mllm_kernel/tensor.hpp>
#include <mllm_kernel/utils.hpp>
#include <tvm/ffi/container/tensor.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "../csrc/my_kernel.cpp"
#include <hwy/foreach_target.h>

#include <mllm_kernel/cpu/my_kernel.hpp>

#if HWY_ONCE
#include <hwy/targets.cc>
#endif

namespace mllm_kernel::cpu {
#if HWY_ONCE

HWY_EXPORT(my_kernel_1);

template <int Constant>
void my_kernel(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) {
  using namespace mllm_kernel::host;
  SymbolicSize N{"num_elements"};
  SymbolicDevice device_;
  (void)TensorMatcher({N})
      .with_dtype<float>()
      .with_device<kDLCPU>(device_)
      .verify(dst)
      .verify(src);
  const size_t n = N.unwrap();
  auto* dst_ptr = static_cast<float*>(dst.data_ptr());
  const auto* src_ptr = static_cast<const float*>(src.data_ptr());
  HWY_DYNAMIC_DISPATCH(my_kernel_1)(dst_ptr, src_ptr, n);
}

// Explicit instantiation
template void my_kernel<1>(tvm::ffi::TensorView, tvm::ffi::TensorView);

#endif
}  // namespace mllm_kernel::cpu
```

### Step 3: Write the Python JIT wrapper

Create `mllm_kernel/cpu/jit/my_kernel.py`:

```python
import torch
from mllm_kernel.jit_utils import jit

@jit(
    args=1,
    device="cpu",
    cpp_files=["my_kernel.cpp"],
    cpp_wrappers=[("my_kernel", "mllm_kernel::cpu::my_kernel<1>")],
    func_name="my_kernel",
)
def _kernel_1(compiled_module, dst, src):
    compiled_module.my_kernel(dst, src)

def my_kernel(src: torch.Tensor) -> torch.Tensor:
    dst = torch.empty_like(src)
    _kernel_1(dst, src)
    return dst
```

**Key CPU differences from CUDA:**

| Aspect | CUDA | CPU |
|--------|------|-----|
| Source file | `.cuh` in `cuda/csrc/` | `.cpp` + `.hpp` in `cpu/csrc/` and `cpu/include/` |
| Namespace | Anonymous `namespace {}` | `mllm_kernel::cpu` |
| Device check | `with_device<kDLCUDA>` | `with_device<kDLCPU>` |
| Launch | `LaunchKernel(grid, block, device)(...)` | Direct function call via `HWY_DYNAMIC_DISPATCH` |
| SIMD | CUDA warps | Highway `ScalableTag<T>` |
| Wrapper fields | `cuda_files`, `cuda_wrappers` | `cpp_files`, `cpp_wrappers` |
| Wrapper name | `"MyKernel::run"` | `"mllm_kernel::cpu::my_kernel<1>"` (fully qualified) |

---

## TensorMatcher Reference

`TensorMatcher` validates shape, dtype, device, and strides of `tvm::ffi::TensorView` arguments.

```cpp
using namespace mllm_kernel::host;

// Symbolic dimensions -- bind on first .verify(), check consistency on subsequent calls
SymbolicSize B{"batch"}, N{"seq_len"}, D{"dim"};
SymbolicSize Stride0{"stride0"};
SymbolicDType dtype;
SymbolicDevice device;

// Shape [B, N, D], contiguous, float32, on CUDA
(void)TensorMatcher({B, N, D})
    .with_dtype<float>(dtype)
    .with_device<kDLCUDA>(device)
    .verify(tensor_a);

// Shape [B, N, D], same dtype and device (already bound)
(void)TensorMatcher({B, N, D})
    .with_dtype(dtype)
    .with_device(device)
    .verify(tensor_b);

// Shape [B, D] with explicit strides (non-contiguous OK)
(void)TensorMatcher({B, D})
    .with_strides({Stride0, 1})
    .with_dtype<int32_t>()
    .with_device(device)
    .verify(indices);

// Multiple acceptable dtypes
SymbolicDType flex_dtype;
(void)TensorMatcher({N})
    .with_dtype<float, __half, __nv_bfloat16>(flex_dtype)
    .with_device(device)
    .verify(mixed_tensor);

// Extract bound values
int64_t batch = B.unwrap();
int64_t dim   = D.unwrap();
DLDevice dev  = device.unwrap();
```

---

## LaunchKernel Reference

```cpp
using namespace mllm_kernel::host;

// Basic launch (resolves CUDA stream from DLDevice)
DLDevice dev = device.unwrap();
LaunchKernel(grid_dim, block_dim, dev)(kernel_func, param_struct);

// With shared memory
LaunchKernel(grid, block, dev, shared_mem_bytes)(kernel, params);

// With PDL (Programmatic Dependent Launch, sm_90+)
LaunchKernel(grid, block, dev).enable_pdl(true)(kernel, params);
```

---

## Utility Reference (`mllm_kernel::host`)

| Function | Description |
|----------|-------------|
| `RuntimeCheck(cond, msg...)` | Throws `PanicError` if `cond` is false |
| `Panic(msg...)` | Always throws (unreachable code) |
| `div_ceil(a, b)` | Integer ceiling division |
| `dtype_bytes(DLDataType)` | Byte size of a DLPack dtype |

CUDA-only (`mllm_kernel::device`):

| Symbol | Value |
|--------|-------|
| `kWarpThreads` | 32 |
| `kFullMask` | 0xffffffff |
| `fp16_t` | `__half` |
| `bf16_t` | `__nv_bfloat16` |

---

## Testing Pattern

Create `tests/test_my_kernel.py`:

```python
import pytest
import torch
from mllm_kernel.cuda.jit.my_kernel import my_kernel

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n", [1, 128, 1024, 65536])
def test_my_kernel(n):
    x = torch.randn(n, dtype=torch.float32, device="cuda")
    result = my_kernel(x)
    torch.cuda.synchronize()
    expected = x * 2.0
    assert torch.allclose(result, expected)
```

Run:
```bash
pytest tests/test_my_kernel.py -v
```

---

## Benchmark Pattern

Create `benchmarks/bench_my_kernel.py`. Use `torch.profiler.profile` with `ProfilerActivity.CPU` and `ProfilerActivity.CUDA`. Compare the JIT kernel against a naive PyTorch implementation and report speedup.

Run:
```bash
python benchmarks/bench_my_kernel.py --num-elements 1000000
```

---

## Checklist for a New Kernel

- [ ] `.cuh` / `.cpp` + `.hpp` kernel source created
- [ ] `TensorMatcher` validates all tensor arguments (shape, dtype, device)
- [ ] No host-side dereference of device pointers
- [ ] Python `@jit` wrapper created with correct `cuda_wrappers` or `cpp_wrappers`
- [ ] Public API function added (allocates output, calls internal `_kernel`)
- [ ] Exported in `jit/__init__.py`
- [ ] JIT cache cleared after `.cuh` edits (`rm -rf ~/.cache/mllm_kernel/cuda_<name>*`)
- [ ] Pytest test with `@pytest.mark.parametrize` and PyTorch reference
- [ ] Benchmark with `torch.profiler` (optional but recommended)

---

## Common Pitfalls

1. **Segfault from dereferencing device pointer on host** -- `tensor.data_ptr()` returns a GPU pointer for CUDA tensors. Never read its contents in host code. Use `TensorMatcher` for validation instead.
2. **Stale JIT cache** -- After editing `.cuh`, delete `~/.cache/mllm_kernel/cuda_<kernel_name>*/`. The old `.so` will be reused otherwise.
3. **Missing `#include <hwy/targets.cc>`** -- CPU kernels must include this inside `#if HWY_ONCE` to provide `GetChosenTarget` for the JIT-built module.
4. **`#pragma once` in Highway `.hpp`** -- Highway's `foreach_target` includes the file multiple times for different SIMD targets. `#pragma once` breaks this.
5. **Wrong wrapper name** -- CUDA uses short names (`"MyKernel::run"`); CPU uses fully qualified names (`"mllm_kernel::cpu::my_kernel<1>"`).
6. **Generator device mismatch in tests** -- `torch.randperm` needs a CUDA generator on CUDA; `torch.randint` only accepts CPU generators. Use separate generators.
