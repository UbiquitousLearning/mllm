// Fused GDN (Gated Delta Net) decode kernel for linear attention.
//
// Performs a single-token recurrent update per request:
//   g     = -exp(A_log) * softplus(a + dt_bias)
//   beta  = sigmoid(b)
//   q     = L2norm(q) * scale
//   k     = L2norm(k)
//   state *= exp(g)                     (decay)
//   v_delta = v - state @ k             (delta rule)
//   v_delta *= beta                     (gated update)
//   state += v_delta outer k            (state update)
//   output = state @ q                  (readout)
//
// Works on SM80+ (Ampere, Jetson Orin, Hopper, ...).
// Matches the algorithm of sglang's fused_sigmoid_gating_delta_rule_update.
//
// Grid : (NV, bs * HV)   where NV = ceil(V / BV)
// Block: BLOCK_K threads  (one thread per K-dimension element)
//
// Each thread owns BV state elements at its K position.
// Two cross-thread reductions (over K) compute delta and output dot products.

#pragma once

#include <mllm_kernel/tensor.hpp>
#include <mllm_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace GDNDecodeKernel {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr int BV = 32;  // V-dimension tile size

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
__device__ __forceinline__ float to_float<__half>(__half val) {
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
__device__ __forceinline__ __half from_float<__half>(float val) {
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
// Block-level scalar reduction (sum across all threads → broadcast result)
// ---------------------------------------------------------------------------

// Reduces a scalar across all threads in the block.
// Returns the sum in ALL threads (via shared memory broadcast).
// smem must have at least (blockDim.x / 32) floats.
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    val = warp_reduce_sum(val);
    if (lane_id == 0) smem[warp_id] = val;
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

// ---------------------------------------------------------------------------
// Block-level vector reduction: BV independent sums across all K threads
// ---------------------------------------------------------------------------

// Each thread contributes partial[0..BV-1].  After this call, the results
// are written to out[0..BV-1] and are valid in all threads.
// reduce_buf must have at least BV * num_warps floats.
// broadcast_buf must have at least BV floats.
__device__ __forceinline__ void block_reduce_bv(
    float partial[BV],
    float* reduce_buf,  // [num_warps * BV]
    float* broadcast_buf, // [BV]
    float out[BV]
) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    // Intra-warp reduction for each bv
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        float val = warp_reduce_sum(partial[bv]);
        if (lane_id == 0) {
            reduce_buf[warp_id * BV + bv] = val;
        }
    }
    __syncthreads();

    // Inter-warp reduction: threads 0..BV-1 each reduce one bv
    if (threadIdx.x < BV) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int w = 0; w < num_warps; w++) {
            sum += reduce_buf[w * BV + threadIdx.x];
        }
        broadcast_buf[threadIdx.x] = sum;
    }
    __syncthreads();

    // Broadcast to all threads
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        out[bv] = broadcast_buf[bv];
    }
}

// ---------------------------------------------------------------------------
// Main GDN decode kernel
// ---------------------------------------------------------------------------

template <typename T, int BLOCK_K>
__global__ void gdn_decode_kernel(
    const T* __restrict__ q_ptr,            // [bs, H, K]
    const T* __restrict__ k_ptr,            // [bs, H, K]
    const T* __restrict__ v_ptr,            // [bs, HV, V]
    const T* __restrict__ a_ptr,            // [bs, HV]
    const T* __restrict__ b_ptr,            // [bs, HV]
    const float* __restrict__ A_log_ptr,    // [HV]
    const float* __restrict__ dt_bias_ptr,  // [HV]
    float* __restrict__ state_pool,         // [pool_size, HV, V, K]
    const int64_t* __restrict__ cache_indices, // [bs]
    T* __restrict__ output_ptr,             // [bs, HV, V]
    const int bs,
    const int H,        // num_k_heads
    const int HV,       // num_v_heads
    const int K,        // head_k_dim
    const int V,        // head_v_dim
    const float scale   // K^-0.5
) {
    // Block indices
    const int bv_block = blockIdx.x;           // V-tile index
    const int batch_head = blockIdx.y;         // batch * HV
    const int i_n = batch_head / HV;           // batch index
    const int i_hv = batch_head % HV;          // value head index
    const int i_h = i_hv * H / HV;            // key head index (GQA mapping)
    const int k_idx = threadIdx.x;             // K-dimension index
    const int v_start = bv_block * BV;         // V-dimension start

    if (i_n >= bs) return;

    // Shared memory layout (declared dynamically)
    extern __shared__ float smem[];
    const int num_warps = BLOCK_K / 32;
    float* sq            = smem;                          // [BLOCK_K]
    float* sk            = smem + BLOCK_K;                // [BLOCK_K]
    float* sv_broadcast  = smem + 2 * BLOCK_K;            // [BV]
    float* warp_buf      = smem + 2 * BLOCK_K + BV;       // [num_warps]
    float* reduce_buf    = smem + 2 * BLOCK_K + BV + num_warps; // [BV * num_warps]

    // ===== 1. Load gating parameters and compute decay + beta =====
    // All threads load the same scalars (cheap, avoids shared memory)
    const float A_log_val = A_log_ptr[i_hv];
    const float dt_bias_val = dt_bias_ptr[i_hv];
    const float a_val = to_float(a_ptr[i_n * HV + i_hv]);
    const float b_val = to_float(b_ptr[i_n * HV + i_hv]);

    const float x = a_val + dt_bias_val;
    // softplus with numerical stability: softplus(x) = log(1+exp(x)), or x for x>20
    const float softplus_x = (x <= 20.0f) ? logf(1.0f + expf(x)) : x;
    const float g = -expf(A_log_val) * softplus_x;
    const float decay = expf(g);
    const float beta = 1.0f / (1.0f + expf(-b_val));

    // ===== 2. Load q, k and compute L2 norms =====
    float q_val = 0.0f, k_val = 0.0f;
    if (k_idx < K) {
        q_val = to_float(q_ptr[i_n * H * K + i_h * K + k_idx]);
        k_val = to_float(k_ptr[i_n * H * K + i_h * K + k_idx]);
    }

    // L2 norm: reduce q*q and k*k across block
    float q_sq_sum = block_reduce_sum(q_val * q_val, warp_buf);
    float k_sq_sum = block_reduce_sum(k_val * k_val, warp_buf);

    float q_norm = rsqrtf(q_sq_sum + 1e-6f);
    float k_norm = rsqrtf(k_sq_sum + 1e-6f);

    // Store normalized q (scaled) and k in shared memory
    if (k_idx < K) {
        sq[k_idx] = q_val * q_norm * scale;
        sk[k_idx] = k_val * k_norm;
    } else {
        sq[k_idx] = 0.0f;
        sk[k_idx] = 0.0f;
    }
    __syncthreads();

    // ===== 3. Load state elements for this thread =====
    const int64_t pool_idx = cache_indices[i_n];
    // state_pool layout: [pool_size, HV, V, K]
    const int64_t state_base = pool_idx * HV * V * K + i_hv * V * K;

    float state[BV];
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        const int v_idx = v_start + bv;
        if (v_idx < V && k_idx < K) {
            state[bv] = state_pool[state_base + (int64_t)v_idx * K + k_idx];
        } else {
            state[bv] = 0.0f;
        }
    }

    // ===== 4. Decay: state *= exp(g) =====
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        state[bv] *= decay;
    }

    // ===== 5. Delta: v_delta[bv] = v[bv] - sum_k(state[bv,k] * k_norm[k]) =====
    float partial_delta[BV];
    const float my_k = sk[k_idx];
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        partial_delta[bv] = state[bv] * my_k;
    }

    float delta[BV];
    block_reduce_bv(partial_delta, reduce_buf, sv_broadcast, delta);

    // Compute v_delta = (v - delta) * beta and broadcast to all threads.
    // Threads 0..BV-1 each load one v element, compute v_delta, write to smem.
    if (k_idx < BV) {
        const int my_v_idx = v_start + k_idx;
        float my_v = (my_v_idx < V)
            ? to_float(v_ptr[i_n * HV * V + i_hv * V + my_v_idx])
            : 0.0f;
        sv_broadcast[k_idx] = (my_v - delta[k_idx]) * beta;
    }
    __syncthreads();

    float v_delta[BV];
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        v_delta[bv] = sv_broadcast[bv];
    }

    // ===== 6. State update: state[bv,k] += v_delta[bv] * k_norm[k] =====
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        state[bv] += v_delta[bv] * my_k;
    }

    // ===== 7. Output: o[bv] = sum_k(state[bv,k] * q_norm_scaled[k]) =====
    float partial_out[BV];
    const float my_q = sq[k_idx];
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        partial_out[bv] = state[bv] * my_q;
    }

    float out_vals[BV];
    block_reduce_bv(partial_out, reduce_buf, sv_broadcast, out_vals);

    // ===== 8. Store output =====
    // output layout: [bs, HV, V]
    if (k_idx < BV) {
        const int v_idx = v_start + k_idx;
        if (v_idx < V) {
            output_ptr[i_n * HV * V + i_hv * V + v_idx] = from_float<T>(out_vals[k_idx]);
        }
    }

    // ===== 9. Store state back to pool =====
    #pragma unroll
    for (int bv = 0; bv < BV; bv++) {
        const int v_idx = v_start + bv;
        if (v_idx < V && k_idx < K) {
            state_pool[state_base + (int64_t)v_idx * K + k_idx] = state[bv];
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrapper (called via TVM FFI)
// ---------------------------------------------------------------------------

void run(
    tvm::ffi::TensorView q,              // [bs, H, K]
    tvm::ffi::TensorView k,              // [bs, H, K]
    tvm::ffi::TensorView v,              // [bs, HV, V]
    tvm::ffi::TensorView a,              // [bs, HV]
    tvm::ffi::TensorView b,              // [bs, HV]
    tvm::ffi::TensorView A_log,          // [HV]
    tvm::ffi::TensorView dt_bias,        // [HV]
    tvm::ffi::TensorView state_pool,     // [pool_size, HV, V, K]
    tvm::ffi::TensorView cache_indices,  // [bs]
    tvm::ffi::TensorView output          // [bs, HV, V]
) {
    using namespace mllm_kernel::host;

    // --- Extract dimensions ---
    auto BS  = SymbolicSize{"bs"};
    auto H_  = SymbolicSize{"H"};
    auto HV_ = SymbolicSize{"HV"};
    auto K_  = SymbolicSize{"K"};
    auto V_  = SymbolicSize{"V"};
    auto PS  = SymbolicSize{"pool_size"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    dtype.set_options<fp16_t, bf16_t>();

    (void)TensorMatcher({BS, H_, K_}).with_dtype(dtype).with_device(device).verify(q);
    (void)TensorMatcher({BS, H_, K_}).with_dtype(dtype).with_device(device).verify(k);
    (void)TensorMatcher({BS, HV_, V_}).with_dtype(dtype).with_device(device).verify(v);
    (void)TensorMatcher({BS, HV_}).with_dtype(dtype).with_device(device).verify(a);
    (void)TensorMatcher({BS, HV_}).with_dtype(dtype).with_device(device).verify(b);
    (void)TensorMatcher({HV_}).with_dtype<float>().with_device(device).verify(A_log);
    (void)TensorMatcher({HV_}).with_dtype<float>().with_device(device).verify(dt_bias);
    (void)TensorMatcher({PS, HV_, V_, K_}).with_dtype<float>().with_device(device).verify(state_pool);
    (void)TensorMatcher({BS}).with_device(device).verify(cache_indices);
    (void)TensorMatcher({BS, HV_, V_}).with_dtype(dtype).with_device(device).verify(output);

    const int bs   = static_cast<int>(BS.unwrap());
    const int H    = static_cast<int>(H_.unwrap());
    const int HV   = static_cast<int>(HV_.unwrap());
    const int K    = static_cast<int>(K_.unwrap());
    const int V    = static_cast<int>(V_.unwrap());
    const float scale = 1.0f / sqrtf(static_cast<float>(K));

    // Block size = K (rounded up to warp multiple, max 1024)
    int block_k = ((K + 31) / 32) * 32;
    if (block_k > 1024) block_k = 1024;
    const int num_warps = block_k / 32;

    // Grid
    const int NV = (V + BV - 1) / BV;
    dim3 grid(NV, bs * HV);
    dim3 block(block_k);

    // Dynamic shared memory: sq[block_k] + sk[block_k] + sv[BV] + warp_buf[nw] + reduce[BV*nw]
    const size_t smem_bytes = (2 * block_k + BV + num_warps + BV * num_warps) * sizeof(float);

    const DLDevice dl_device = device.unwrap();

    // Typed launch helper
    #define LAUNCH_GDN_DECODE(CType, BKVAL)                                     \
        LaunchKernel(grid, block, dl_device, smem_bytes)(                        \
            gdn_decode_kernel<CType, BKVAL>,                                    \
            static_cast<const CType*>(q.data_ptr()),                            \
            static_cast<const CType*>(k.data_ptr()),                            \
            static_cast<const CType*>(v.data_ptr()),                            \
            static_cast<const CType*>(a.data_ptr()),                            \
            static_cast<const CType*>(b.data_ptr()),                            \
            static_cast<const float*>(A_log.data_ptr()),                        \
            static_cast<const float*>(dt_bias.data_ptr()),                      \
            static_cast<float*>(state_pool.data_ptr()),                         \
            static_cast<const int64_t*>(cache_indices.data_ptr()),              \
            static_cast<CType*>(output.data_ptr()),                             \
            bs, H, HV, K, V, scale                                             \
        )

    // Dispatch based on dtype and block size
    if (dtype.is_type<bf16_t>()) {
        if      (block_k == 64)  { LAUNCH_GDN_DECODE(__nv_bfloat16, 64);  }
        else if (block_k == 128) { LAUNCH_GDN_DECODE(__nv_bfloat16, 128); }
        else if (block_k == 256) { LAUNCH_GDN_DECODE(__nv_bfloat16, 256); }
        else                     { LAUNCH_GDN_DECODE(__nv_bfloat16, 256); }
    } else {
        if      (block_k == 64)  { LAUNCH_GDN_DECODE(__half, 64);  }
        else if (block_k == 128) { LAUNCH_GDN_DECODE(__half, 128); }
        else if (block_k == 256) { LAUNCH_GDN_DECODE(__half, 256); }
        else                     { LAUNCH_GDN_DECODE(__half, 256); }
    }

    #undef LAUNCH_GDN_DECODE
}

}  // namespace GDNDecodeKernel
