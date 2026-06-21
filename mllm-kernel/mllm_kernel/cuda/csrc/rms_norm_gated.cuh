// Fused RMSNorm with optional SiLU gating for Qwen3.5 GDN attention.
//
// Computes: output = rmsnorm(x, weight, eps) * silu(z)   (if z provided)
//           output = rmsnorm(x, weight, eps)              (if z is null)
//
// Where: rmsnorm(x) = x / sqrt(mean(x^2) + eps) * weight
//        silu(z) = z * sigmoid(z)
//
// This kernel fuses both operations into a single pass over the data,
// maximizing memory bandwidth utilization.  Each block processes one row
// (one token position).
//
// Supported dtypes: float16, bfloat16 (accumulation in float32).

#pragma once

#include <mllm_kernel/tensor.hpp>
#include <mllm_kernel/utils.hpp>
#include <mllm_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace RMSNormGatedKernel {

// ---------------------------------------------------------------------------
// Warp-level reduction
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---------------------------------------------------------------------------
// Type conversion helpers
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<half>(half val) {
    return __half2float(val);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <>
__device__ __forceinline__ float to_float<float>(float val) {
    return val;
}

template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ half from_float<half>(float val) {
    return __float2half(val);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

template <>
__device__ __forceinline__ float from_float<float>(float val) {
    return val;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_SIZE>
__global__ void rms_norm_gated_kernel(
    T* __restrict__ output,           // [M, N]
    const T* __restrict__ input,      // [M, N]
    const T* __restrict__ weight,     // [N]
    const T* __restrict__ gate,       // [M, N] or nullptr
    const int M,                      // number of rows
    const int N,                      // number of columns (hidden_size)
    const float eps
) {
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const T* x_row = input + row * N;
    T* out_row = output + row * N;
    const T* z_row = (gate != nullptr) ? gate + row * N : nullptr;

    // --- Pass 1: compute sum of squares ---
    float sum_sq = 0.0f;
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float val = to_float(x_row[col]);
        sum_sq += val * val;
    }

    // Block-level reduction
    __shared__ float shared_sum[32];  // one per warp
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    sum_sq = warp_reduce_sum(sum_sq);
    if (lane_id == 0) {
        shared_sum[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? shared_sum[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_sum[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(shared_sum[0] / (float)N + eps);

    // --- Pass 2: normalize, scale by weight, optionally gate with silu(z) ---
    for (int col = tid; col < N; col += BLOCK_SIZE) {
        float val = to_float(x_row[col]);
        float w = to_float(weight[col]);

        float normed = val * rms * w;

        if (z_row != nullptr) {
            float z = to_float(z_row[col]);
            // silu(z) = z * sigmoid(z)
            float silu_z = z / (1.0f + expf(-z));
            normed *= silu_z;
        }

        out_row[col] = from_float<T>(normed);
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper (called via TVM FFI)
// ---------------------------------------------------------------------------

void run(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView gate,      // empty tensor (numel==0) means no gate
    double eps
) {
    using namespace mllm_kernel::host;

    auto M = SymbolicSize{"M"};
    auto N = SymbolicSize{"N"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    dtype.set_options<fp16_t, bf16_t, float>();

    (void)TensorMatcher({M, N}).with_dtype(dtype).with_device(device).verify(input);
    (void)TensorMatcher({M, N}).with_dtype(dtype).with_device(device).verify(output);
    (void)TensorMatcher({N}).with_dtype(dtype).with_device(device).verify(weight);

    const int rows = static_cast<int>(M.unwrap());
    const int cols = static_cast<int>(N.unwrap());
    const bool has_gate = (gate.numel() > 0);

    constexpr int BLOCK_SIZE = 256;

    if (dtype.is_type<fp16_t>()) {
        LaunchKernel(rows, BLOCK_SIZE, device.unwrap())(
            rms_norm_gated_kernel<half, BLOCK_SIZE>,
            static_cast<half*>(output.data_ptr()),
            static_cast<const half*>(input.data_ptr()),
            static_cast<const half*>(weight.data_ptr()),
            has_gate ? static_cast<const half*>(gate.data_ptr()) : nullptr,
            rows, cols, static_cast<float>(eps)
        );
    } else if (dtype.is_type<bf16_t>()) {
        LaunchKernel(rows, BLOCK_SIZE, device.unwrap())(
            rms_norm_gated_kernel<__nv_bfloat16, BLOCK_SIZE>,
            static_cast<__nv_bfloat16*>(output.data_ptr()),
            static_cast<const __nv_bfloat16*>(input.data_ptr()),
            static_cast<const __nv_bfloat16*>(weight.data_ptr()),
            has_gate ? static_cast<const __nv_bfloat16*>(gate.data_ptr()) : nullptr,
            rows, cols, static_cast<float>(eps)
        );
    } else {
        LaunchKernel(rows, BLOCK_SIZE, device.unwrap())(
            rms_norm_gated_kernel<float, BLOCK_SIZE>,
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(input.data_ptr()),
            static_cast<const float*>(weight.data_ptr()),
            has_gate ? static_cast<const float*>(gate.data_ptr()) : nullptr,
            rows, cols, static_cast<float>(eps)
        );
    }
}

}  // namespace RMSNormGatedKernel
