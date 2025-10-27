#ifndef MLLM_GEMM_AARCH64_HPP
#define MLLM_GEMM_AARCH64_HPP

#include <cstddef>
#include <cstdint>
// using namespace mllm;

// Quantization
void quantize_q8_0_4x4(const float *__restrict x, void *__restrict y, int64_t k);
void quantize_q8_0_4x8(const float *__restrict x, void *__restrict y, int64_t k);

void quantize_mat_q8_0(const float *__restrict x, void *__restrict y, int64_t nrows, int64_t n_per_row, int64_t blck_size_interleave);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
size_t quantize_q4_0_4x4(const float *__restrict src, void *__restrict dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
size_t quantize_q4_0_4x8(const float *__restrict src, void *__restrict dst, int64_t nrows, int64_t n_per_row, const float *imatrix);
size_t quantize_q4_0_8x8(const float *__restrict src, void *__restrict dst, int64_t nrows, int64_t n_per_row, const float *imatrix);

//===----------------------------------------------------------------------===//
// GEMV
//===----------------------------------------------------------------------===//
void gemv_q4_0_4x4_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias = nullptr);
void gemv_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias = nullptr);
void gemv_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias = nullptr);

// NOTE: Do not add a bias flag in gemv_q4_0_4x4_q8_0. It may cause branch miss hit problem.
void _gemv_q4_0_4x4_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias);
void _gemv_q4_0_4x8_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias);
void _gemv_q4_0_8x8_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias);

//===----------------------------------------------------------------------===//
// GEMM
//===----------------------------------------------------------------------===//
void gemm_q4_0_4x4_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias = nullptr);
void gemm_q4_0_4x8_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias = nullptr);
void gemm_q4_0_8x8_q8_0(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias = nullptr);
void _gemm_q4_0_4x4_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias);
void _gemm_q4_0_4x8_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias);
void _gemm_q4_0_8x8_q8_0_bias(int n, float *__restrict s, size_t bs, const void *__restrict vx, const void *__restrict vy, int nr, int nc, const void *__restrict bias);

void quantize_row_q4_0_4x4(const float *__restrict x, void *__restrict y, int k);
void quantize_row_q4_0_4x4(const float *__restrict x, void *__restrict y, int k, int raw);

#endif // MLLM_GEMM_HPP