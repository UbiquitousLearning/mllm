// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>
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

namespace {
struct MMShapeKey {
  int m, n, k;
  int kind;  // 0=sgemm/sgemm-like, 1=gemm, 2=gemv
  bool operator==(const MMShapeKey& o) const {
    return m == o.m && n == o.n && k == o.k && kind == o.kind;
  }
};
struct MMShapeKeyHash {
  size_t operator()(const MMShapeKey& s) const {
    // 简单 hash（够用）
    size_t h = (size_t)s.m * 1315423911u;
    h ^= (size_t)s.n * 2654435761u;
    h ^= (size_t)s.k * 97531u;
    h ^= (size_t)s.kind;
    return h;
  }
};
struct MMAgg {
  uint64_t calls = 0;   // 逻辑调用次数（gemv 用“行数”计）
  uint64_t flops = 0;   // 估算 flops：2*m*n*k（gemv 用 m=1, calls=rows）
};

static bool g_mm_shape_on = false;
static std::once_flag g_mm_shape_init;
static std::mutex g_mm_shape_mu;
static std::unordered_map<MMShapeKey, MMAgg, MMShapeKeyHash> g_mm_shape;

static void mm_shape_init() {
  const char* e = std::getenv("MLLM_MATMUL_SHAPE_LOG");
  g_mm_shape_on = (e && *e && *e != '0');
  if (!g_mm_shape_on) return;

  std::atexit([] {
    std::vector<std::pair<MMShapeKey, MMAgg>> v;
    {
      std::lock_guard<std::mutex> lk(g_mm_shape_mu);
      v.reserve(g_mm_shape.size());
      for (auto& kv : g_mm_shape) v.push_back(kv);
    }
    std::sort(v.begin(), v.end(),
              [](auto& a, auto& b) { return a.second.flops > b.second.flops; });

    std::fprintf(stderr, "\n[MLLM_MATMUL_SHAPE_LOG] top shapes by FLOPs:\n");
    int limit = (int)std::min<size_t>(30, v.size());
    for (int i = 0; i < limit; ++i) {
      const auto& k = v[i].first;
      const auto& a = v[i].second;
      const char* kind =
          (k.kind == 0 ? "sgemm" : (k.kind == 1 ? "gemm" : "gemv"));
      std::fprintf(stderr,
                   "  %2d) %-4s M=%d N=%d K=%d  calls=%llu  flops=%.3e\n",
                   i + 1, kind, k.m, k.n, k.k,
                   (unsigned long long)a.calls,
                   (double)a.flops);
    }
    std::fprintf(stderr, "\n");
  });
}

static inline void mm_shape_record(int kind, int m, int n, int k,
                                   uint64_t calls, uint64_t flops) {
  std::call_once(g_mm_shape_init, mm_shape_init);
  if (!g_mm_shape_on) return;
  std::lock_guard<std::mutex> lk(g_mm_shape_mu);
  MMAgg& a = g_mm_shape[MMShapeKey{m, n, k, kind}];
  a.calls += calls;
  a.flops += flops;
}
}  // namespace

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

  // NOTE: batch_count_mm is ONLY for MLLM_MATMUL_SHAPE_LOG aggregation (does not affect compute path)
int64_t batch_count_mm = 1;
  for (size_t i = 0; i + 2 < dst_shape.size(); ++i) batch_count_mm *= dst_shape[i];
  mm_shape_record(0, M, N, K,
                  (uint64_t)batch_count_mm,
                  (uint64_t)(2.0 * (double)batch_count_mm * M * N * (double)K));


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
        if (id == 0) {
          // flops = 2*M*N*K（注意 llamafile_sgemm 参数顺序是 N,M,K/... 但数学等价）
          mm_shape_record(/*kind=*/0, (int)M, (int)N, (int)K,
                          /*calls=*/1,
                          /*flops=*/2ull * (uint64_t)M * (uint64_t)N * (uint64_t)K);
        }
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
    // 统计“逻辑工作量”：gemm 覆盖 M4 行；gemv 覆盖剩余行（每行一次 gemv）
    const int64_t m_gemm = ((gemm != nullptr) && (M > 3)) ? (M - M % 4) : 0;
    const int64_t m_gemv = M - m_gemm;

    if (m_gemm > 0) {
      mm_shape_record(/*kind=*/1, (int)m_gemm, (int)N, (int)K,
                      /*calls=*/1,
                      /*flops=*/2ull * (uint64_t)m_gemm * (uint64_t)N * (uint64_t)K);
    }
    if (m_gemv > 0) {
      // gemv 每行一次：用 calls=m_gemv，shape 记成 M=1
      mm_shape_record(/*kind=*/2, /*m=*/1, (int)N, (int)K,
                      /*calls=*/(uint64_t)m_gemv,
                      /*flops=*/2ull * (uint64_t)m_gemv * (uint64_t)N * (uint64_t)K);
    }

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
