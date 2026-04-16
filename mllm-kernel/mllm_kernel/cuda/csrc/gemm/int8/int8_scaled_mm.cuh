#pragma once

#include <mllm_kernel/tensor.hpp>
#include <mllm_kernel/utils.hpp>
#include <mllm_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

template<typename scalar_t>
__device__ inline float to_float(scalar_t v);

template<>
__device__ inline float to_float<fp16_t>(fp16_t v) {
  return __half2float(v);
}

template<>
__device__ inline float to_float<bf16_t>(bf16_t v) {
  return __bfloat162float(v);
}

template<typename scalar_t>
__device__ inline scalar_t from_float(float v);

template<>
__device__ inline fp16_t from_float<fp16_t>(float v) {
  return __float2half_rn(v);
}

template<>
__device__ inline bf16_t from_float<bf16_t>(float v) {
  return __float2bfloat16(v);
}

template<typename scalar_t>
__global__ void int8_scaled_mm_kernel(
    const int8_t* __restrict__ mat_a,
    const int8_t* __restrict__ mat_b,
    const float* __restrict__ scales_a,
    const float* __restrict__ scales_b,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldo,
    bool has_bias) {
  const int64_t row = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) {
    return;
  }

  int32_t acc = 0;
  const int8_t* a_row = mat_a + row * lda;
  for (int64_t k = 0; k < K; ++k) {
    acc += static_cast<int32_t>(a_row[k]) * static_cast<int32_t>(mat_b[k * ldb + col]);
  }

  float value = static_cast<float>(acc) * scales_a[row] * scales_b[col];
  if (has_bias) {
    value += to_float<scalar_t>(bias[col]);
  }
  out[row * ldo + col] = from_float<scalar_t>(value);
}

}  // namespace

template<typename scalar_t>
void int8_scaled_mm(
    tvm::ffi::TensorView mat_a,
    tvm::ffi::TensorView mat_b,
    tvm::ffi::TensorView scales_a,
    tvm::ffi::TensorView scales_b,
    tvm::ffi::TensorView bias,
    tvm::ffi::TensorView out) {
  using namespace mllm_kernel::host;

  SymbolicSize M{"M"};
  SymbolicSize K{"K"};
  SymbolicSize N{"N"};
  SymbolicSize lda{"lda"};
  SymbolicSize ldb{"ldb"};
  SymbolicSize ldo{"ldo"};
  SymbolicDevice device;

  TensorMatcher({M, K})
      .with_strides({lda, 1})
      .with_dtype<int8_t>()
      .with_device<kDLCUDA>(device)
      .verify(mat_a);

  TensorMatcher({K, N})
      .with_strides({ldb, 1})
      .with_dtype<int8_t>()
      .with_device(device)
      .verify(mat_b);

  TensorMatcher({M})
      .with_dtype<float>()
      .with_device(device)
      .verify(scales_a);

  TensorMatcher({N})
      .with_dtype<float>()
      .with_device(device)
      .verify(scales_b);

  TensorMatcher({M, N})
      .with_strides({ldo, 1})
      .with_dtype<scalar_t>()
      .with_device(device)
      .verify(out);

  SymbolicSize bias_len{"bias_len"};
  TensorMatcher({bias_len})
      .with_dtype<scalar_t>()
      .with_device(device)
      .verify(bias);

  const int64_t m = M.unwrap();
  const int64_t n = N.unwrap();
  const int64_t k = K.unwrap();
  RuntimeCheck(m >= 0 && n >= 0 && k >= 0, "Negative matrix sizes are not allowed");
  if (m == 0 || n == 0 || k == 0) {
    return;
  }

  const int64_t bias_numel = bias_len.unwrap();
  const bool has_bias = bias_numel > 0;
  RuntimeCheck(
      bias_numel == 0 || bias_numel == n,
      "bias must be empty or have shape [N], got bias_len=",
      bias_numel,
      ", N=",
      n);

  const dim3 block_dim(16, 16);
  const dim3 grid_dim(div_ceil(n, static_cast<int64_t>(block_dim.x)),
                      div_ceil(m, static_cast<int64_t>(block_dim.y)));

  LaunchKernel(grid_dim, block_dim, device.unwrap())(
      int8_scaled_mm_kernel<scalar_t>,
      static_cast<const int8_t*>(mat_a.data_ptr()),
      static_cast<const int8_t*>(mat_b.data_ptr()),
      static_cast<const float*>(scales_a.data_ptr()),
      static_cast<const float*>(scales_b.data_ptr()),
      has_bias ? static_cast<const scalar_t*>(bias.data_ptr()) : nullptr,
      static_cast<scalar_t*>(out.data_ptr()),
      m,
      n,
      k,
      lda.unwrap(),
      ldb.unwrap(),
      ldo.unwrap(),
      has_bias);
}
