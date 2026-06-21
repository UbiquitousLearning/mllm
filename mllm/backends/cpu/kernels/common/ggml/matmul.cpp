// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <cassert>
#include <cstdint>

#include "mllm/core/Parallel.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/matmul.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/vec_dot_type.hpp"
#include "mllm/backends/cpu/kernels/common/llamafile/llamafile_sgemm.hpp"

// For arithmetic functions
void mllm_add_fp32(float* a, float* b, float* c, int n) {
  int i = 0;
#if defined(__ARM_NEON)
  for (i = 0; i <= n - 4; i += 4) {
    float32x4_t vec_a = vld1q_f32(&a[i]);
    float32x4_t vec_b = vld1q_f32(&b[i]);
    float32x4_t vec_c = vaddq_f32(vec_a, vec_b);
    vst1q_f32(&c[i], vec_c);
  }
#elif defined(__AVX2__) || defined(__AVX__)
  for (i = 0; i <= n - 8; i += 8) {
    __m256 vec_a = _mm256_loadu_ps(&a[i]);
    __m256 vec_b = _mm256_loadu_ps(&b[i]);
    __m256 vec_c = _mm256_add_ps(vec_a, vec_b);
    _mm256_storeu_ps(&c[i], vec_c);
  }
#endif
  for (; i < n; i++) { c[i] = a[i] + b[i]; }
}

namespace mllm::cpu::ggml {

#define LLAMAFILE_SGEMM

/**
 * NOTE: This only considered Linear operation with gguf quantizated weights, where weight is a 2D matrix
 * As Matmul Op is performed in float data type. It is recommended using other implementations
 */
void mat_mul(const Tensor& src0_, const Tensor& src1, Tensor& dst, bool support_bias, Tensor* bias, bool transpose0,
             bool transpose1, int thread_count) {
  // src1 = W  src0 = x
  // transpose0=false  transpose1=true

  // Get shapes
  auto src0_shape = src0_.shape();
  auto src1_shape = src1.shape();
  auto dst_shape = dst.shape();

  // For 2D tensors: [M, K] x [K, N] -> [M, N]
  // For higher dimensional: [..., M, K] x [K, N] -> [..., M, N]
  const int M = transpose0 ? src0_shape[src0_shape.size() - 1] : src0_shape[src0_shape.size() - 2];
  const int K = transpose0 ? src0_shape[src0_shape.size() - 2] : src0_shape[src0_shape.size() - 1];
  const int N = transpose1 ? src1_shape[src1_shape.size() - 2] : src1_shape[src1_shape.size() - 1];

  auto src0_dtype = src0_.dtype();
  auto src1_dtype = src1.dtype();
  auto dst_dtype = dst.dtype();

  auto src1_type_size = bytesOfType(src1_dtype);
  auto src0_type_size = bytesOfType(src0_dtype);
  auto src0_blck_size = lanesOfType(src0_dtype);
  auto src1_blck_size = lanesOfType(src1_dtype);

  // Calculate batch count for higher dimensional tensors
  int batch_count = 1;
  for (size_t i = 0; i < src0_shape.size() - 2; ++i) { batch_count *= src0_shape[i]; }

#ifdef LLAMAFILE_SGEMM
  // Try llamafile path first

  // Use llamafile implementation for default case
  // Linear operation: output = input * weight^T + bias
  // In llamafile convention: C = A * B^T, where A is input, B is weight
  const int ld_src0 = K;  // Leading dimension of input
  const int ld_src1 = K;  // Leading dimension of weight (weight is stored as [out_channels, in_channels])
  const int ld_dst = N;   // Leading dimension of output

  void* bias_ptr = bias ? bias->ptr<void>() : nullptr;
  DataTypes bias_type = bias ? bias->dtype() : MLLM_TYPE_F32;

  if (!check_llamafile_sgemm(N, M, K / src0_blck_size, src1_dtype, src0_dtype, dst_dtype, ld_src1, ld_src0 / src0_blck_size,
                             ld_dst)) {
    goto llamafile_fallback;
  }
  MLLM_RT_ASSERT(dst.dtype() == MLLM_TYPE_F32);  // only support output as fp32
  if (batch_count == 1) {
    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, id, 0, thread_count, 1, {
      if (!llamafile_sgemm(N, M, K / src0_blck_size, src1.ptr<mllm_byte_t>(), ld_src1 / src1_blck_size,  // B matrix (weight)
                           src0_.ptr<mllm_byte_t>(), ld_src0 / src0_blck_size,                           // A matrix (input)
                           dst.ptr<mllm_byte_t>(), ld_dst, id, thread_count, src1_dtype, src0_dtype, dst_dtype, bias_ptr,
                           bias_type)) {
        MLLM_WARN("LlamaFile sgemm failed");
      }
    });
  } else {
    // Handle batched operation
    auto src0_stride = src0_.stride();
    auto src1_stride = src1.stride();
    auto dst_stride = dst.stride();

    for (int b = 0; b < batch_count; ++b) {
      MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, id, 0, thread_count, 1, {
        size_t src0_offset = b * src0_stride[src0_stride.size() - 3];
        size_t src1_offset =
            (src1_stride.size() > 2 && src1_stride[src1_stride.size() - 3] > 0) ? b * src1_stride[src1_stride.size() - 3] : 0;
        size_t dst_offset = b * dst_stride[dst_stride.size() - 3];

        if (!llamafile_sgemm(N, M, K / src0_blck_size, src1.ptr<mllm_byte_t>() + src1_offset * src1_type_size / src1_blck_size,
                             ld_src1 / src1_blck_size,  // B matrix (weight)
                             src0_.ptr<mllm_byte_t>() + src0_offset * src0_type_size / src0_blck_size,
                             ld_src0 / src0_blck_size,  // A matrix (input)
                             dst.ptr<mllm_byte_t>() + dst_offset * bytesOfType(dst_dtype), ld_dst, id, thread_count, src1_dtype,
                             src0_dtype, dst_dtype, bias_ptr, bias_type)) {
          MLLM_WARN("LlamaFile sgemm failed");
        }
      });
    }
  }
  return;

#endif

llamafile_fallback:;
  // Get type traits for src1
  auto src1_type_trait = getRuntimeTypeTraits(src1_dtype);
  auto vec_dot_type = src1_type_trait.vec_dot_type;
  mllm_vec_dot_func vec_dot = src1_type_trait.vec_dot;
  mllm_gemv_func gemv = src1_type_trait.gemv;
  mllm_gemm_func gemm = src1_type_trait.gemm;
  int blck_size_interleave = src1_type_trait.blck_size_interleave;

  // Get type traits for vec_dot_type
  auto vec_dot_type_trait = getRuntimeTypeTraits(vec_dot_type);
  mllm_from_float_func x_to_vec_dot_type = vec_dot_type_trait.from_float;
  mllm_from_float_to_mat_func from_float_to_mat = vec_dot_type_trait.from_float_to_mat;

  // Convert src0 to vec_dot_type if needed
  auto not_vec_dot_type = src0_dtype != vec_dot_type;
  Tensor quantized_src0;        // temporary tensor for conversion
  const Tensor* src0 = &src0_;  // NOTE: after quantization, should not use src0_

  auto src0_stride = src0_.stride();
  auto src1_stride = src1.stride();
  auto dst_stride = dst.stride();

  if (not_vec_dot_type) {
    // convert x.dtype to vec_dot_type
    // so that we can use vec_dot to calculate dot product
    assert(src0_dtype == MLLM_TYPE_F32);  // x should be fp32
    quantized_src0 = Tensor::empty(src0_.shape(), vec_dot_type, src0_.device()).alloc();
    auto to_stride = quantized_src0.stride();

    int64_t i_processed = 0;
    if ((from_float_to_mat != nullptr) && (gemv != nullptr) && src0_shape.size() >= 2) {
      for (int b = 0; b < batch_count; b++) {
        MLLM_CONDITIONAL_PARALLEL_FOR(
            thread_count > 1, thread_count, s, 0, src0_shape[src0_shape.size() - 2] - src0_shape[src0_shape.size() - 2] % 4, 4,
            {
              size_t src0_offset = b * src0_stride[src0_stride.size() - 3] + s * src0_stride[src0_stride.size() - 2];
              size_t to_offset = b * to_stride[to_stride.size() - 3] + s * to_stride[to_stride.size() - 2];

              from_float_to_mat(
                  (src0_.ptr<float>() + src0_offset),
                  (quantized_src0.ptr<mllm_byte_t>() + to_offset / lanesOfType(vec_dot_type) * bytesOfType(vec_dot_type)), 4,
                  src0->shape().back(), blck_size_interleave);
            });
        i_processed = src0_shape[src0_shape.size() - 2] - src0_shape[src0_shape.size() - 2] % 4;
      }
    }

    // Handle remaining elements
    auto total_iter = batch_count * (src0_shape[src0_shape.size() - 2] - i_processed);
    MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, combined, 0, total_iter, 1, {
      int total_remaining = src0_shape[src0_shape.size() - 2] - i_processed;

      int b = combined / total_remaining;
      int s = combined % total_remaining + i_processed;

      size_t src0_offset = (b > 0 ? src0_stride[src0_stride.size() - 3] * b : 0) + s * src0_stride[src0_stride.size() - 2];
      size_t to_offset = (b > 0 ? to_stride[to_stride.size() - 3] * b : 0) + s * to_stride[to_stride.size() - 2];

      x_to_vec_dot_type(src0_.ptr<float>() + src0_offset,
                        (quantized_src0.ptr<mllm_byte_t>() + to_offset / lanesOfType(vec_dot_type) * bytesOfType(vec_dot_type)),
                        src0_shape.back());
    });

    src0 = &quantized_src0;
    src0_dtype = src0->dtype();
    src0_type_size = bytesOfType(src0->dtype());
    src0_blck_size = lanesOfType(src0->dtype());
  }

  // After quantize, try llamafile path first
#ifdef LLAMAFILE_SGEMM
  if (check_llamafile_sgemm(N, M, K, src1_dtype, src0_dtype, dst_dtype, ld_src1, ld_src0, ld_dst)) {
    if (batch_count == 1) {
      MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, id, 0, thread_count, 1, {
        if (!llamafile_sgemm(N, M, K / src0_blck_size, src1.ptr<mllm_byte_t>(), ld_src1 / src1_blck_size,  // B matrix (weight)
                             src0->ptr<mllm_byte_t>(), ld_src0 / src0_blck_size,                           // A matrix (input)
                             dst.ptr<mllm_byte_t>(), ld_dst, id, thread_count, src1_dtype, src0_dtype, dst_dtype, bias_ptr,
                             bias_type)) {
          MLLM_WARN("LlamaFile sgemm failed");
        }
      });
    } else {
      // Handle batched operation
      const auto src0_stride = src0->stride();
      const auto src1_stride = src1.stride();
      const auto dst_stride = dst.stride();

      for (int b = 0; b < batch_count; ++b) {
        MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, id, 0, thread_count, 1, {
          size_t src0_offset = b * src0_stride[src0_stride.size() - 3];
          size_t src1_offset =
              (src1_stride.size() > 2 && src1_stride[src1_stride.size() - 3] > 0) ? b * src1_stride[src1_stride.size() - 3] : 0;
          size_t dst_offset = b * dst_stride[dst_stride.size() - 3];

          if (!llamafile_sgemm(N, M, K / src0_blck_size,
                               src1.ptr<mllm_byte_t>() + src1_offset * src1_type_size / src1_blck_size,
                               ld_src1 / src1_blck_size,  // B matrix (weight)
                               src0->ptr<mllm_byte_t>() + src0_offset * src0_type_size / src0_blck_size,
                               ld_src0 / src0_blck_size,  // A matrix (input)
                               dst.ptr<mllm_byte_t>() + dst_offset * bytesOfType(dst_dtype), ld_dst, id, thread_count,
                               src1_dtype, src0_dtype, dst_dtype, bias_ptr, bias_type)) {
            MLLM_WARN("LlamaFile sgemm failed");
          }
        });
      }
    }
    return;  // sgemm done
  }
#endif  // LLAMAFILE_SGEMM

  // Use gemv/gemm if available
  if ((gemv != nullptr) && dst_dtype == MLLM_TYPE_F32) {
    int nth = thread_count;

    if (!support_bias) {
      MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, ith, 0, nth, 1, {
        int64_t i_processed = 0;
        int64_t seq_start = (ith * N) / nth;
        int64_t seq_end = ((ith + 1) * N) / nth;

        if ((gemm != nullptr) && (M > 3)) {
          gemm(K, dst.ptr<float>() + seq_start * N, N,
               src1.ptr<mllm_byte_t>() + seq_start * N * src1_type_size / src1_blck_size, (const char*)src0->ptr<mllm_byte_t>(),
               M - M % 4, N / nth, /*bias=*/nullptr);
          i_processed = M - M % 4;
        }

        for (int iter = i_processed; iter < M; iter++) {
          gemv(K, dst.ptr<float>() + iter * N + seq_start, N,
               src1.ptr<mllm_byte_t>() + seq_start * N * src1_type_size / src1_blck_size,
               src0->ptr<mllm_byte_t>() + iter * K * src0_type_size / src0_blck_size, 1, N / nth, /*bias=*/nullptr);
        }
      });
    } else {
      MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, ith, 0, nth, 1, {
        int64_t i_processed = 0;
        int64_t seq_start = (ith * N) / nth;
        int64_t seq_end = ((ith + 1) * N) / nth;

        if ((gemm != nullptr) && (M > 3)) {
          gemm(K, dst.ptr<float>() + seq_start, N,
               reinterpret_cast<const char*>(src1.ptr<mllm_byte_t>() + seq_start * N * src1_type_size / src1_blck_size),
               reinterpret_cast<const char*>(src0->ptr<mllm_byte_t>()), M - M % 4, N / nth, /*bias=*/nullptr);
          i_processed = M - M % 4;
        }

        for (int iter = i_processed; iter < M; iter++) {
          gemv(K, dst.ptr<float>() + iter * N + seq_start, N,
               reinterpret_cast<const char*>(src1.ptr<mllm_byte_t>() + seq_start * N * src1_type_size / src1_blck_size),
               reinterpret_cast<const char*>(src0->ptr<mllm_byte_t>() + iter * K * src0_type_size / src0_blck_size), 1, N / nth,
               /*bias=*/nullptr);
        }
      });

      // Add bias for all batches, heads and sequences
      MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, combined, 0, batch_count * M, 1, {
        int b = combined / M;
        int s = combined % M;

        size_t dst_offset =
            (batch_count > 1 ? b * dst_stride[dst_stride.size() - 3] : 0) + s * dst_stride[dst_stride.size() - 2];

        mllm_add_fp32(dst.ptr<float>() + dst_offset, bias->ptr<float>(), dst.ptr<float>() + dst_offset, N);
      });
    }
    return;
  }  // end of gemv/gemm path

  // Fallback to vec_dot implementation
  const int64_t blck_0 = 16;

  MLLM_CONDITIONAL_PARALLEL_FOR(thread_count > 1, thread_count, combined, 0, batch_count * M * ((N / blck_0) + 1), 1, {
    int temp = combined;
    int b = temp / (M * ((N / blck_0) + 1));
    temp %= (M * ((N / blck_0) + 1));
    int m = temp / ((N / blck_0) + 1);
    int block = temp % ((N / blck_0) + 1);

    for (int n = block * blck_0; n < (block + 1) * blck_0 && n < N; n++) {
      if (vec_dot) {
        int s_1;
        int d_1;
        int s_0;
        int d_0;
        if (!transpose0 && transpose1) {
          s_1 = n;
          d_1 = 0;
          s_0 = m;
          d_0 = 0;
        } else if (!transpose0 && !transpose1) {
          s_1 = 0;
          d_1 = n;
          s_0 = m;
          d_0 = 0;
        } else {
          s_1 = 0;
          d_1 = n;
          s_0 = 0;
          d_0 = m;
        }

        float sumf = 0.0f;

        // Calculate src1 and src0 offsets based on transpose flags
        size_t src0_offset =
            (batch_count > 1 ? b * src0_stride[src0_stride.size() - 3] : 0) + s_0 * src0_stride[src0_stride.size() - 2] + d_0;
        size_t src1_offset = s_1 * src1_stride[src1_stride.size() - 2] + d_1;

        const char* src0_ptr =
            reinterpret_cast<const char*>(src0->ptr<mllm_byte_t>() + src0_offset * src0_type_size / src0_blck_size);
        const char* src1_ptr =
            reinterpret_cast<const char*>(src1.ptr<mllm_byte_t>() + src1_offset * src1_type_size / src1_blck_size);

        vec_dot(K, &sumf, src1_ptr, src0_ptr);

        if (support_bias && bias) { sumf += bias->ptr<float>()[n]; }

        // Calculate dst offset
        size_t dst_offset = 0;
        if (dst_stride.size() > 2) dst_offset += b * dst_stride[dst_stride.size() - 3];
        dst_offset += m * N + n;

        dst.ptr<float>()[dst_offset] = sumf;
      }
    }
  });
}

}  // namespace mllm::cpu::ggml
