// High-performance CUDA kernel to build FlashInfer KV index arrays from
// pymllm's ReqToTokenPool mapping table.
//
// This is the CUDA-C equivalent of the Triton kernel
// `_create_kv_indices_triton` previously defined in
// `pymllm/layers/attention/flashinfer_backend.py`.
//
// Motivation
// ----------
// FlashInfer's paged KV attention API expects a *flat* buffer of KV indices
// (`kv_indices`) together with a prefix-sum pointer array (`kv_indptr`).
//
//   * `kv_indices` is a 1-D int32 array that stores, for every token of every
//     sequence in a batch, the corresponding *slot index* in the KV cache.
//   * `kv_indptr` (length = batch_size + 1) stores prefix sums over the
//     per-sequence token counts.  For sequence `i` we have tokens in:
//
//         kv_indices[kv_indptr[i] : kv_indptr[i + 1]]
//
// In pymllm, the mapping from (request_slot, position_in_sequence) to KV slot
// index is stored in a 2-D tensor `req_to_token` owned by `ReqToTokenPool`:
//
//     req_to_token[req_slot, position] -> kv_index (int32)
//
// For each batch we also know:
//   * which request slots we are serving: `req_pool_indices[bs]`
//   * how many tokens to use from each sequence: `page_kernel_lens[bs]`
//   * the starting position inside each sequence: `kv_start_idx[bs]` (optional,
//     used for sliding-window / partial-context attention)
//
// This kernel converts that 2-D layout into the flat `(kv_indptr, kv_indices)`
// layout in a single, highly parallel CUDA pass:
//
//   For each sequence i in the batch:
//     - let req = req_pool_indices[i]
//     - let len = page_kernel_lens[i]
//     - let start = kv_start_idx[i] (or 0 if not provided)
//     - let offset = kv_indptr[i]
//     - for j in [0, len):
//         kv_indices[offset + j] = req_to_token[req, start + j]
//
// Requirements / invariants
// -------------------------
// * `req_to_token` is int32 (aligned with sglang).
// * All tensors must reside on the same CUDA device.
// * The kernel is designed for extremely high throughput:
//     - a block is assigned per sequence (batch element),
//     - threads cooperate within the block to copy the token range with
//       coalesced loads/stores.
// * Shape and dtype checks are performed at runtime via mllm_kernel's
//   TensorMatcher utilities, so misuse is caught with clear error messages.
//
// Integration
// -----------
// The exported entry point is `CreateKvIndicesKernel::run(...)`.  The Python
// wrapper in `mllm_kernel/cuda/jit/create_kv_indices.py` JIT-compiles this
// kernel and exposes a `create_kv_indices(...)` function which is then called
// by `pymllm.layers.attention.flashinfer_backend`.

#pragma once

#include <mllm_kernel/tensor.hpp>  // TensorMatcher, SymbolicSize, SymbolicDevice, SymbolicDType
#include <mllm_kernel/utils.hpp>   // div_ceil, RuntimeCheck, Panic
#include <mllm_kernel/utils.cuh>   // LaunchKernel

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

// ---------------------------------------------------------------------------
// Parameter block passed to the CUDA kernel
// ---------------------------------------------------------------------------
//
// We keep this struct trivially-copyable so it can be passed via
// `__grid_constant__` if desired.  Each field is carefully documented to make
// the data flow explicit.

struct CreateKvIndicesParams {
  // Pointer to ReqToTokenPool mapping table:
  //   req_to_token[req_slot, position] -> kv_index (int32)
  // shape: [max_reqs, max_context_len]
  const int32_t* __restrict__ req_to_token;

  // Request slots participating in this batch.
  // shape: [batch_size]
  const int32_t* __restrict__ req_pool_indices;

  // Number of tokens to copy for each sequence in the batch.
  // shape: [batch_size]
  const int32_t* __restrict__ page_kernel_lens;

  // Prefix sums over per-sequence token counts.
  //   kv_indptr[i] is the starting offset in kv_indices for sequence i.
  // shape: [batch_size + 1]
  const int32_t* __restrict__ kv_indptr;

  // Optional starting position inside each request's sequence.  When nullptr,
  // we assume start = 0 for all sequences.  When non-null, shape is
  // [batch_size].
  const int32_t* __restrict__ kv_start_idx;

  // Output flat KV index buffer (int32).  Length must be at least
  // kv_indptr[batch_size].
  int32_t* __restrict__ kv_indices;

  // Stride of the first dimension of req_to_token, i.e. the number of
  // positions per request (max_context_len).
  int32_t req_to_token_stride;

  // Number of sequences in the batch.
  uint32_t batch_size;

  // Whether kv_start_idx is valid (1) or should be ignored (0).
  uint32_t has_kv_start;
};

// We use a fixed block size chosen to balance occupancy and per-sequence
// parallelism.  Each block is mapped to a single sequence and threads within
// the block cooperate to copy its token range.
constexpr int kBlockSize = 256;

// ---------------------------------------------------------------------------
// Core CUDA kernel
// ---------------------------------------------------------------------------
//
// Grid mapping:
//   * blockIdx.x -> sequence index `i` in [0, batch_size)
//   * threadIdx.x -> intra-sequence worker; threads stride over the token
//     range [0, len) with step `blockDim.x`.
//
// This design has several advantages:
//   * No inter-block synchronisation is required.
//   * Memory accesses are fully coalesced because each thread block walks a
//     contiguous segment of the `req_to_token` and `kv_indices` arrays.
//   * It handles variable-length sequences naturally; sequences with more
//     tokens simply iterate more in the inner loop.

__global__ void create_kv_indices_kernel(const CreateKvIndicesParams params) {
  const uint32_t seq_id = blockIdx.x;  // which sequence in the batch
  if (seq_id >= params.batch_size) { return; }

  // Resolve the request slot for this sequence.
  const int32_t req_slot = params.req_pool_indices[seq_id];

  // Compute the output range [out_offset, out_offset + len) in kv_indices.
  const int32_t out_offset = params.kv_indptr[seq_id];
  const int32_t len = params.page_kernel_lens[seq_id];

  // Compute the starting position inside the original sequence.
  int32_t start = 0;
  if (params.has_kv_start && params.kv_start_idx != nullptr) { start = params.kv_start_idx[seq_id]; }

  // Base pointers for this sequence.
  const int32_t* __restrict__ row = params.req_to_token + static_cast<int64_t>(req_slot) * params.req_to_token_stride;
  int32_t* __restrict__ out = params.kv_indices + out_offset;

  // Each thread in the block handles a strided subset of [0, len).
  for (int32_t t = threadIdx.x; t < len; t += blockDim.x) {
    // Guard against out-of-bounds reads if (start + t) exceeds the
    // configured context length.  Under normal conditions upstream
    // invariants guarantee `start + len <= req_to_token_stride`, but
    // this check makes the kernel robust against misconfigured inputs
    // and prevents rare segmentation faults observed during testing.
    const int32_t pos = start + t;
    if (pos < 0 || pos >= params.req_to_token_stride) { continue; }

    out[t] = row[pos];
  }
}

// ---------------------------------------------------------------------------
// Host-side launcher used by the JIT wrapper
// ---------------------------------------------------------------------------
//
// `CreateKvIndicesKernel::run(...)` is the C++ entry point that will be bound
// to a TVM FFI function and called from Python via the JIT utility.  It is
// responsible for:
//   1. Validating tensor shapes / dtypes / devices.
//   2. Extracting symbolic sizes and strides.
//   3. Building the parameter block.
//   4. Launching the CUDA kernel using mllm_kernel::host::LaunchKernel.

struct CreateKvIndicesKernel {
  static void run(tvm::ffi::TensorView req_to_token, tvm::ffi::TensorView req_pool_indices,
                  tvm::ffi::TensorView page_kernel_lens, tvm::ffi::TensorView kv_indptr, tvm::ffi::TensorView kv_start_idx,
                  tvm::ffi::TensorView kv_indices) {
    using namespace mllm_kernel::host;

    // ---------------------------------------------------------------------
    // 1. Validate input tensors
    // ---------------------------------------------------------------------
    // req_to_token: [max_reqs, max_context_len], int32, CUDA
    SymbolicSize MaxReqs{"max_reqs"};
    SymbolicSize MaxCtx{"max_context_len"};
    SymbolicSize ReqStride{"req_stride"};
    SymbolicDType req_dtype;
    SymbolicDevice device;

    (void)TensorMatcher({MaxReqs, MaxCtx})
        .with_strides({ReqStride, 1})
        .with_dtype<int32_t>(req_dtype)
        .with_device<kDLCUDA>(device)
        .verify(req_to_token);

    // req_pool_indices: [B], int32, CUDA
    SymbolicSize B{"batch_size"};
    SymbolicSize ReqPoolStride{"req_pool_stride"};
    (void)TensorMatcher({B}).with_strides({ReqPoolStride}).with_dtype<int32_t>().with_device(device).verify(req_pool_indices);

    // page_kernel_lens: [B], int32, same device
    SymbolicSize PageStride{"page_stride"};
    (void)TensorMatcher({B}).with_strides({PageStride}).with_dtype<int32_t>().with_device(device).verify(page_kernel_lens);

    // kv_indptr: [Nind], int32, same device (we later require Nind >= B + 1)
    SymbolicSize Nind{"indptr_len"};
    (void)TensorMatcher({Nind}).with_dtype<int32_t>().with_device(device).verify(kv_indptr);

    // kv_start_idx: either [B] or [0]; int32, same device
    SymbolicSize StartLen{"start_len"};
    SymbolicSize StartStride{"start_stride"};
    (void)TensorMatcher({StartLen}).with_strides({StartStride}).with_dtype<int32_t>().with_device(device).verify(kv_start_idx);

    // kv_indices: [Nidx], int32, same device
    SymbolicSize Nidx{"num_indices"};
    (void)TensorMatcher({Nidx}).with_dtype<int32_t>().with_device(device).verify(kv_indices);

    // Extract concrete sizes.
    const int64_t batch_size = B.unwrap();
    const int64_t indptr_len = Nind.unwrap();
    const int64_t req_stride = ReqStride.unwrap();

    // Basic consistency checks.
    RuntimeCheck(batch_size > 0, "batch_size must be positive, got ", batch_size);
    RuntimeCheck(indptr_len >= batch_size + 1, "kv_indptr length (", indptr_len, ") must be at least batch_size+1 (",
                 batch_size + 1, ")");

    // NOTE: We intentionally do NOT read kv_indptr[batch_size] on the host to
    // validate that kv_indices is large enough.  kv_indptr resides in device
    // memory and dereferencing it from host code would be an illegal memory
    // access (segfault).  Callers are responsible for ensuring that
    // kv_indices.numel() >= kv_indptr[batch_size].

    // kv_start_idx is optional; when StartLen == 0 we treat it as absent.
    RuntimeCheck(StartLen.unwrap() == 0 || StartLen.unwrap() == batch_size,
                 "kv_start_idx must have length 0 or batch_size; got ", StartLen.unwrap(), " vs batch_size=", batch_size);

    const bool has_kv_start = (StartLen.unwrap() == batch_size);

    // ---------------------------------------------------------------------
    // 2. Build parameter block
    // ---------------------------------------------------------------------
    CreateKvIndicesParams params{
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .req_pool_indices = static_cast<const int32_t*>(req_pool_indices.data_ptr()),
        .page_kernel_lens = static_cast<const int32_t*>(page_kernel_lens.data_ptr()),
        .kv_indptr = static_cast<const int32_t*>(kv_indptr.data_ptr()),
        .kv_start_idx = has_kv_start ? static_cast<const int32_t*>(kv_start_idx.data_ptr()) : nullptr,
        .kv_indices = static_cast<int32_t*>(kv_indices.data_ptr()),
        .req_to_token_stride = static_cast<int32_t>(req_stride),
        .batch_size = static_cast<uint32_t>(batch_size),
        .has_kv_start = has_kv_start ? 1u : 0u,
    };

    const DLDevice dl_device = device.unwrap();

    // ---------------------------------------------------------------------
    // 3. Launch the CUDA kernel
    // ---------------------------------------------------------------------
    // We launch one block per sequence so that each sequence can be processed
    // independently with fully coalesced memory accesses.  The per-thread
    // inner loop runs over the token range [0, len) with stride = blockDim.x.

    const int grid_size = static_cast<int>(batch_size);

    LaunchKernel(grid_size, kBlockSize, dl_device)(create_kv_indices_kernel, params);
  }
};

}  // namespace
