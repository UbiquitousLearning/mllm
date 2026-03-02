// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Embedding kernels migrated from TensorRT-Edge-LLM.
// Reference: https://github.com/NVIDIA/TensorRT-Edge-LLM/tree/main/cpp/kernels/embeddingKernels
//
// Supported operations:
//   1. embedding_lookup             — standard token embedding
//   2. embedding_lookup_with_image  — token + image embedding fusion
//   3. assemble_deepstack_embedding — extract image-only embeddings
//   4. embedding_lookup_multimodal  — text + image + audio embedding

#pragma once

#include <mllm_kernel/tensor.hpp>
#include <mllm_kernel/utils.hpp>
#include <mllm_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstddef>
#include <cstdint>

namespace {

// ───────────────────────────────────────────────────────────────
// Constants
// ───────────────────────────────────────────────────────────────

constexpr uint32_t kEmbNumWarps = 4;
constexpr uint32_t kEmbBlockSize = kEmbNumWarps * device::kWarpThreads;

// ───────────────────────────────────────────────────────────────
// Vectorised warp-level row copy / zero
// ───────────────────────────────────────────────────────────────

namespace detail {

template<typename Vec>
__device__ __forceinline__ void warp_copy_row(const void* __restrict__ src,
                                              void* __restrict__ dst,
                                              int64_t num_vecs) {
  const int lane = threadIdx.x % device::kWarpThreads;
  const auto* __restrict__ s = static_cast<const Vec*>(src);
  auto* __restrict__ d = static_cast<Vec*>(dst);
  for (int64_t i = lane; i < num_vecs; i += device::kWarpThreads) {
    d[i] = s[i];
  }
}

template<typename Vec>
__device__ __forceinline__ void warp_zero_row(void* __restrict__ dst,
                                              int64_t num_vecs) {
  const int lane = threadIdx.x % device::kWarpThreads;
  auto* __restrict__ d = static_cast<Vec*>(dst);
  const Vec zero{};
  for (int64_t i = lane; i < num_vecs; i += device::kWarpThreads) {
    d[i] = zero;
  }
}

}  // namespace detail

__device__ __forceinline__ void copy_or_zero_row(const void* __restrict__ src,
                                                 void* __restrict__ dst,
                                                 int64_t row_bytes) {
  if (row_bytes % 16 == 0) {
    const int64_t n = row_bytes / 16;
    if (src) detail::warp_copy_row<uint4>(src, dst, n);
    else     detail::warp_zero_row<uint4>(dst, n);
  } else if (row_bytes % 8 == 0) {
    const int64_t n = row_bytes / 8;
    if (src) detail::warp_copy_row<uint2>(src, dst, n);
    else     detail::warp_zero_row<uint2>(dst, n);
  } else {
    const int64_t n = row_bytes / 4;
    if (src) detail::warp_copy_row<uint32_t>(src, dst, n);
    else     detail::warp_zero_row<uint32_t>(dst, n);
  }
}

// ───────────────────────────────────────────────────────────────
// Parameter blocks (passed via __grid_constant__)
// ───────────────────────────────────────────────────────────────

struct EmbeddingLookupParams {
  void* __restrict__ output;
  const void* __restrict__ input_ids;
  const void* __restrict__ embedding_table;
  int64_t num_tokens;
  int64_t stride_bytes;
  int32_t vocab_size;
};

struct EmbeddingLookupWithImageParams {
  void* __restrict__ output;
  const void* __restrict__ input_ids;
  const void* __restrict__ embedding_table;
  const void* __restrict__ image_embeds;
  int64_t num_tokens;
  int64_t stride_bytes;
  int32_t vocab_size;
  int64_t image_token_len;
};

struct AssembleDeepstackParams {
  void* __restrict__ output;
  const void* __restrict__ input_ids;
  const void* __restrict__ deepstack_features;
  int64_t num_tokens;
  int64_t stride_bytes;
  int32_t vocab_size;
  int64_t num_image_tokens;
};

struct EmbeddingMultimodalParams {
  void* __restrict__ output;
  const void* __restrict__ input_ids;
  const void* __restrict__ embedding_table;
  const void* __restrict__ multimodal_indices;
  const void* __restrict__ image_embeds;
  const void* __restrict__ audio_embeds;
  int64_t num_tokens;
  int64_t stride_bytes;
  int32_t vocab_size;
  int32_t image_token_id;
  int64_t image_token_len;
  int32_t audio_token_id;
  int64_t audio_token_len;
};

// ───────────────────────────────────────────────────────────────
// Kernel 1: Standard Embedding Lookup
// ───────────────────────────────────────────────────────────────

__global__ void embedding_lookup_kernel(
    const __grid_constant__ EmbeddingLookupParams params) {
  const uint32_t warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (warp_id >= params.num_tokens) return;

  const auto token_id = static_cast<const int32_t*>(params.input_ids)[warp_id];

  const void* src = nullptr;
  if (token_id >= 0 && token_id < params.vocab_size) {
    src = device::pointer::offset(params.embedding_table,
                                  static_cast<int64_t>(token_id) * params.stride_bytes);
  }
  auto* dst = device::pointer::offset(params.output,
                                      static_cast<int64_t>(warp_id) * params.stride_bytes);

  copy_or_zero_row(src, dst, params.stride_bytes);
}

// ───────────────────────────────────────────────────────────────
// Kernel 2: Embedding Lookup with Image Insertion
// ───────────────────────────────────────────────────────────────

__global__ void embedding_lookup_with_image_kernel(
    const __grid_constant__ EmbeddingLookupWithImageParams params) {
  const uint32_t warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (warp_id >= params.num_tokens) return;

  const auto token_id = static_cast<const int32_t*>(params.input_ids)[warp_id];

  const void* src = nullptr;
  if (token_id >= params.vocab_size) {
    const int32_t visual_id = token_id - params.vocab_size;
    if (visual_id < params.image_token_len) {
      src = device::pointer::offset(params.image_embeds,
                                    static_cast<int64_t>(visual_id) * params.stride_bytes);
    }
  } else if (token_id >= 0) {
    src = device::pointer::offset(params.embedding_table,
                                  static_cast<int64_t>(token_id) * params.stride_bytes);
  }
  auto* dst = device::pointer::offset(params.output,
                                      static_cast<int64_t>(warp_id) * params.stride_bytes);

  copy_or_zero_row(src, dst, params.stride_bytes);
}

// ───────────────────────────────────────────────────────────────
// Kernel 3: Assemble Deepstack Embedding
// ───────────────────────────────────────────────────────────────

__global__ void assemble_deepstack_embedding_kernel(
    const __grid_constant__ AssembleDeepstackParams params) {
  const uint32_t warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (warp_id >= params.num_tokens) return;

  const auto token_id = static_cast<const int32_t*>(params.input_ids)[warp_id];

  const void* src = nullptr;
  if (token_id >= params.vocab_size) {
    const int32_t ds_idx = token_id - params.vocab_size;
    if (ds_idx < params.num_image_tokens) {
      src = device::pointer::offset(params.deepstack_features,
                                    static_cast<int64_t>(ds_idx) * params.stride_bytes);
    }
  }
  auto* dst = device::pointer::offset(params.output,
                                      static_cast<int64_t>(warp_id) * params.stride_bytes);

  copy_or_zero_row(src, dst, params.stride_bytes);
}

// ───────────────────────────────────────────────────────────────
// Kernel 4: Multimodal Embedding Lookup
// ───────────────────────────────────────────────────────────────

__global__ void embedding_lookup_multimodal_kernel(
    const __grid_constant__ EmbeddingMultimodalParams params) {
  const uint32_t warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (warp_id >= params.num_tokens) return;

  const auto token_id = static_cast<const int32_t*>(params.input_ids)[warp_id];

  const void* src = nullptr;
  if (params.image_embeds != nullptr && token_id == params.image_token_id) {
    const auto idx = static_cast<const int32_t*>(params.multimodal_indices)[warp_id];
    if (idx >= 0 && idx < params.image_token_len) {
      src = device::pointer::offset(params.image_embeds,
                                    static_cast<int64_t>(idx) * params.stride_bytes);
    }
  } else if (params.audio_embeds != nullptr && token_id == params.audio_token_id) {
    const auto idx = static_cast<const int32_t*>(params.multimodal_indices)[warp_id];
    if (idx >= 0 && idx < params.audio_token_len) {
      src = device::pointer::offset(params.audio_embeds,
                                    static_cast<int64_t>(idx) * params.stride_bytes);
    }
  } else if (token_id >= 0 && token_id < params.vocab_size) {
    src = device::pointer::offset(params.embedding_table,
                                  static_cast<int64_t>(token_id) * params.stride_bytes);
  }
  auto* dst = device::pointer::offset(params.output,
                                      static_cast<int64_t>(warp_id) * params.stride_bytes);

  copy_or_zero_row(src, dst, params.stride_bytes);
}

// ───────────────────────────────────────────────────────────────
// Host-side launch wrappers
// ───────────────────────────────────────────────────────────────

void embedding_lookup(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input_ids,
    tvm::ffi::TensorView embedding_table) {
  using namespace mllm_kernel::host;

  auto N = SymbolicSize{"num_tokens"};
  auto V = SymbolicSize{"vocab_size"};
  auto H = SymbolicSize{"hidden_size"};
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  dtype.set_options<fp16_t, bf16_t>();

  (void)TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>().verify(input_ids);
  (void)TensorMatcher({V, H}).with_dtype(dtype).with_device(device).verify(embedding_table);
  (void)TensorMatcher({N, H}).with_dtype(dtype).with_device(device).verify(output);

  const int64_t dtype_size = dtype_bytes(dtype.unwrap());
  const auto num_tokens = static_cast<uint32_t>(N.unwrap());
  const auto stride_bytes = H.unwrap() * dtype_size;

  RuntimeCheck(stride_bytes % 4 == 0,
               "stride_bytes must be at least 4-byte aligned, got ", stride_bytes);

  const auto params = EmbeddingLookupParams{
      .output = output.data_ptr(),
      .input_ids = input_ids.data_ptr(),
      .embedding_table = embedding_table.data_ptr(),
      .num_tokens = static_cast<int64_t>(num_tokens),
      .stride_bytes = stride_bytes,
      .vocab_size = static_cast<int32_t>(V.unwrap()),
  };

  const dim3 block(device::kWarpThreads, kEmbNumWarps);
  const auto grid = div_ceil(num_tokens, kEmbNumWarps);
  LaunchKernel(grid, block, device.unwrap())(embedding_lookup_kernel, params);
}

void embedding_lookup_with_image(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input_ids,
    tvm::ffi::TensorView embedding_table,
    tvm::ffi::TensorView image_embeds) {
  using namespace mllm_kernel::host;

  auto N = SymbolicSize{"num_tokens"};
  auto V = SymbolicSize{"vocab_size"};
  auto H = SymbolicSize{"hidden_size"};
  auto I = SymbolicSize{"image_token_len"};
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  dtype.set_options<fp16_t, bf16_t>();

  (void)TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>().verify(input_ids);
  (void)TensorMatcher({V, H}).with_dtype(dtype).with_device(device).verify(embedding_table);
  (void)TensorMatcher({I, H}).with_dtype(dtype).with_device(device).verify(image_embeds);
  (void)TensorMatcher({N, H}).with_dtype(dtype).with_device(device).verify(output);

  const int64_t dtype_size = dtype_bytes(dtype.unwrap());
  const auto num_tokens = static_cast<uint32_t>(N.unwrap());
  const auto stride_bytes = H.unwrap() * dtype_size;

  RuntimeCheck(stride_bytes % 4 == 0,
               "stride_bytes must be at least 4-byte aligned, got ", stride_bytes);

  const auto params = EmbeddingLookupWithImageParams{
      .output = output.data_ptr(),
      .input_ids = input_ids.data_ptr(),
      .embedding_table = embedding_table.data_ptr(),
      .image_embeds = image_embeds.data_ptr(),
      .num_tokens = static_cast<int64_t>(num_tokens),
      .stride_bytes = stride_bytes,
      .vocab_size = static_cast<int32_t>(V.unwrap()),
      .image_token_len = I.unwrap(),
  };

  const dim3 block(device::kWarpThreads, kEmbNumWarps);
  const auto grid = div_ceil(num_tokens, kEmbNumWarps);
  LaunchKernel(grid, block, device.unwrap())(embedding_lookup_with_image_kernel, params);
}

void assemble_deepstack_embedding(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input_ids,
    tvm::ffi::TensorView deepstack_features,
    int vocab_size) {
  using namespace mllm_kernel::host;

  auto N = SymbolicSize{"num_tokens"};
  auto F = SymbolicSize{"num_image_tokens"};
  auto H = SymbolicSize{"hidden_size"};
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  dtype.set_options<fp16_t, bf16_t>();

  (void)TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>().verify(input_ids);
  (void)TensorMatcher({F, H}).with_dtype(dtype).with_device(device).verify(deepstack_features);
  (void)TensorMatcher({N, H}).with_dtype(dtype).with_device(device).verify(output);

  const int64_t dtype_size = dtype_bytes(dtype.unwrap());
  const auto num_tokens = static_cast<uint32_t>(N.unwrap());
  const auto stride_bytes = H.unwrap() * dtype_size;

  RuntimeCheck(stride_bytes % 4 == 0,
               "stride_bytes must be at least 4-byte aligned, got ", stride_bytes);

  const auto params = AssembleDeepstackParams{
      .output = output.data_ptr(),
      .input_ids = input_ids.data_ptr(),
      .deepstack_features = deepstack_features.data_ptr(),
      .num_tokens = static_cast<int64_t>(num_tokens),
      .stride_bytes = stride_bytes,
      .vocab_size = static_cast<int32_t>(vocab_size),
      .num_image_tokens = F.unwrap(),
  };

  const dim3 block(device::kWarpThreads, kEmbNumWarps);
  const auto grid = div_ceil(num_tokens, kEmbNumWarps);
  LaunchKernel(grid, block, device.unwrap())(assemble_deepstack_embedding_kernel, params);
}

void embedding_lookup_multimodal(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView input_ids,
    tvm::ffi::TensorView embedding_table,
    tvm::ffi::TensorView multimodal_indices,
    tvm::ffi::TensorView image_embeds,
    tvm::ffi::TensorView audio_embeds,
    int image_token_id,
    int audio_token_id) {
  using namespace mllm_kernel::host;

  auto N = SymbolicSize{"num_tokens"};
  auto V = SymbolicSize{"vocab_size"};
  auto H = SymbolicSize{"hidden_size"};
  auto IL = SymbolicSize{"image_token_len"};
  auto AL = SymbolicSize{"audio_token_len"};
  auto dtype = SymbolicDType{};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  dtype.set_options<fp16_t, bf16_t>();

  (void)TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>().verify(input_ids);
  (void)TensorMatcher({V, H}).with_dtype(dtype).with_device(device).verify(embedding_table);
  (void)TensorMatcher({N}).with_dtype<int32_t>().with_device<kDLCUDA>().verify(multimodal_indices);
  (void)TensorMatcher({IL, H}).with_dtype(dtype).with_device(device).verify(image_embeds);
  (void)TensorMatcher({AL, H}).with_dtype(dtype).with_device(device).verify(audio_embeds);
  (void)TensorMatcher({N, H}).with_dtype(dtype).with_device(device).verify(output);

  const int64_t dtype_size = dtype_bytes(dtype.unwrap());
  const auto num_tokens = static_cast<uint32_t>(N.unwrap());
  const auto stride_bytes = H.unwrap() * dtype_size;
  const auto image_token_len = IL.unwrap();
  const auto audio_token_len = AL.unwrap();

  RuntimeCheck(stride_bytes % 4 == 0,
               "stride_bytes must be at least 4-byte aligned, got ", stride_bytes);

  const auto params = EmbeddingMultimodalParams{
      .output = output.data_ptr(),
      .input_ids = input_ids.data_ptr(),
      .embedding_table = embedding_table.data_ptr(),
      .multimodal_indices = multimodal_indices.data_ptr(),
      .image_embeds = (image_token_len > 0) ? image_embeds.data_ptr() : nullptr,
      .audio_embeds = (audio_token_len > 0) ? audio_embeds.data_ptr() : nullptr,
      .num_tokens = static_cast<int64_t>(num_tokens),
      .stride_bytes = stride_bytes,
      .vocab_size = static_cast<int32_t>(V.unwrap()),
      .image_token_id = static_cast<int32_t>(image_token_id),
      .image_token_len = image_token_len,
      .audio_token_id = static_cast<int32_t>(audio_token_id),
      .audio_token_len = audio_token_len,
  };

  const dim3 block(device::kWarpThreads, kEmbNumWarps);
  const auto grid = div_ceil(num_tokens, kEmbNumWarps);
  LaunchKernel(grid, block, device.unwrap())(embedding_lookup_multimodal_kernel, params);
}

}  // namespace