// Copyright SGLang Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Store KV cache kernel: efficiently scatter key/value tensors into a
// pre-allocated KV cache pool using warp-level vectorized copies.
//
// Reference: sglang jit_kernel/csrc/elementwise/kvcache.cuh

#pragma once

#include <mllm_kernel/tensor.hpp>
#include <mllm_kernel/utils.hpp>
#include <mllm_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// ───────────────────────────────────────────────────────────────
// Parameter block passed to the kernel via __grid_constant__
// ───────────────────────────────────────────────────────────────

struct StoreKVCacheParams {
  const void* __restrict__ k;
  const void* __restrict__ v;
  void* __restrict__ k_cache;
  void* __restrict__ v_cache;
  const void* __restrict__ indices;
  int64_t stride_k_bytes;
  int64_t stride_v_bytes;
  int64_t stride_cache_bytes;
  int64_t stride_indices;
  uint32_t batch_size;
};

constexpr uint32_t kNumWarps = 4;
constexpr uint32_t kThreadsPerBlock = kNumWarps * device::kWarpThreads;

// ───────────────────────────────────────────────────────────────
// Vectorized warp-level KV copy
// ───────────────────────────────────────────────────────────────
//
// Each warp copies kElementBytes of K data and kElementBytes of V
// data using the widest possible aligned vector type (uint4 = 16B,
// uint2 = 8B, or uint32_t = 4B).

namespace detail {

template<typename Vec>
__device__ __forceinline__ void warp_copy_bytes(const void* __restrict__ src, void* __restrict__ dst, int64_t num_vecs) {
  const int lane = threadIdx.x % device::kWarpThreads;
  const auto* s = static_cast<const Vec*>(src);
  auto* d = static_cast<Vec*>(dst);
  for (int64_t i = lane; i < num_vecs; i += device::kWarpThreads) { d[i] = s[i]; }
}

}  // namespace detail

template<int64_t kElementBytes>
__device__ __forceinline__ void copy_kv_warp(const void* __restrict__ k_src, const void* __restrict__ v_src,
                                             void* __restrict__ k_dst, void* __restrict__ v_dst) {
  static_assert(kElementBytes > 0 && kElementBytes % 4 == 0, "Element size must be a positive multiple of 4 bytes");

  // Pick the widest aligned vector type the element size supports.
  if constexpr (kElementBytes % 16 == 0) {
    constexpr int64_t N = kElementBytes / 16;
    detail::warp_copy_bytes<uint4>(k_src, k_dst, N);
    detail::warp_copy_bytes<uint4>(v_src, v_dst, N);
  } else if constexpr (kElementBytes % 8 == 0) {
    constexpr int64_t N = kElementBytes / 8;
    detail::warp_copy_bytes<uint2>(k_src, k_dst, N);
    detail::warp_copy_bytes<uint2>(v_src, v_dst, N);
  } else {
    constexpr int64_t N = kElementBytes / 4;
    detail::warp_copy_bytes<uint32_t>(k_src, k_dst, N);
    detail::warp_copy_bytes<uint32_t>(v_src, v_dst, N);
  }
}

// ───────────────────────────────────────────────────────────────
// Main kernel
// ───────────────────────────────────────────────────────────────
//
// Template parameters:
//   kElementBytes  total bytes per token row (head_num * head_dim * dtype_size)
//   kSplit         how many warps collaborate on one element (1, 2, or 4)
//   kUsePDL        whether to emit PDL synchronisation instructions
//   T              index dtype (int32_t or int64_t)

template<int64_t kElementBytes, int kSplit, bool kUsePDL, typename T>
__global__ void store_kvcache(const __grid_constant__ StoreKVCacheParams params) {
  using namespace device;
  constexpr auto kSplitSize = kElementBytes / kSplit;

  const uint32_t warp_id = blockIdx.x * kNumWarps + threadIdx.x / kWarpThreads;
  const uint32_t item_id = warp_id / kSplit;
  const uint32_t split_id = warp_id % kSplit;

  const auto& [k_input, v_input, k_cache, v_cache, indices, stride_k, stride_v, stride_cache, stride_indices, batch_size] =
      params;

  if (item_id >= batch_size) return;

  const auto index_ptr = static_cast<const T*>(indices) + item_id * stride_indices;
  PDLWaitPrimary<kUsePDL>();

  const auto index = *index_ptr;
  const auto k_src = pointer::offset(k_input, item_id * stride_k, split_id * kSplitSize);
  const auto v_src = pointer::offset(v_input, item_id * stride_v, split_id * kSplitSize);
  const auto k_dst = pointer::offset(k_cache, index * stride_cache, split_id * kSplitSize);
  const auto v_dst = pointer::offset(v_cache, index * stride_cache, split_id * kSplitSize);

  copy_kv_warp<kSplitSize>(k_src, v_src, k_dst, v_dst);
  PDLTriggerSecondary<kUsePDL>();
}

template<int64_t kElementBytes, bool kUsePDL>
struct StoreKVCacheKernel {
  static_assert(kElementBytes > 0 && kElementBytes % 4 == 0);

  template<int kSplit, typename T>
  static constexpr auto store_kernel = store_kvcache<kElementBytes, kSplit, kUsePDL, T>;

  template<typename T>
  static auto get_kernel(int num_split) {
    using namespace mllm_kernel::host;
    if constexpr (kElementBytes % (4 * 128) == 0) {
      if (num_split == 4) return store_kernel<4, T>;
    }
    if constexpr (kElementBytes % (2 * 128) == 0) {
      if (num_split == 2) return store_kernel<2, T>;
    }
    if (num_split == 1) return store_kernel<1, T>;
    Panic("Unsupported num_split ", num_split, " for element size ", kElementBytes);
  }

  static void run(tvm::ffi::TensorView k, tvm::ffi::TensorView v, tvm::ffi::TensorView k_cache, tvm::ffi::TensorView v_cache,
                  tvm::ffi::TensorView indices, int num_split) {
    using namespace mllm_kernel::host;

    auto B = SymbolicSize{"batch_size"};
    auto D = SymbolicSize{"element_size"};
    auto KS = SymbolicSize{"k_stride"};
    auto VS = SymbolicSize{"v_stride"};
    auto S = SymbolicSize{"cache_stride"};
    auto I = SymbolicSize{"indices_stride"};
    auto dtype = SymbolicDType{};
    auto device = SymbolicDevice{};
    auto indice_dtype = SymbolicDType{};
    device.set_options<kDLCUDA>();

    // k, v: [B, D]  with strides [KS, 1]
    (void)TensorMatcher({B, D}).with_strides({KS, 1}).with_dtype(dtype).with_device(device).verify(k);
    (void)TensorMatcher({B, D}).with_strides({VS, 1}).with_dtype(dtype).with_device(device).verify(v);

    // k_cache, v_cache: [*, D]  with strides [S, 1]
    (void)TensorMatcher({-1, D}).with_strides({S, 1}).with_dtype(dtype).with_device(device).verify(k_cache).verify(v_cache);

    // indices: [B]  with strides [I]
    (void)TensorMatcher({B}).with_strides({I}).with_dtype<int32_t, int64_t>(indice_dtype).with_device(device).verify(indices);

    const int64_t dtype_size = dtype_bytes(dtype.unwrap());
    const uint32_t num_elements = static_cast<uint32_t>(B.unwrap());
    RuntimeCheck(kElementBytes == dtype_size * D.unwrap(), "Element size mismatch: expected ", kElementBytes, " but got ",
                 dtype_size * D.unwrap());

    const auto params = StoreKVCacheParams{
        .k = k.data_ptr(),
        .v = v.data_ptr(),
        .k_cache = k_cache.data_ptr(),
        .v_cache = v_cache.data_ptr(),
        .indices = indices.data_ptr(),
        .stride_k_bytes = KS.unwrap() * dtype_size,
        .stride_v_bytes = VS.unwrap() * dtype_size,
        .stride_cache_bytes = S.unwrap() * dtype_size,
        .stride_indices = I.unwrap(),
        .batch_size = num_elements,
    };

    const auto use_int32 = indice_dtype.is_type<int32_t>();
    const auto kernel = use_int32 ? get_kernel<int32_t>(num_split) : get_kernel<int64_t>(num_split);
    const auto num_blocks = div_ceil(num_elements * num_split, kNumWarps);

    LaunchKernel(num_blocks, kThreadsPerBlock, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
