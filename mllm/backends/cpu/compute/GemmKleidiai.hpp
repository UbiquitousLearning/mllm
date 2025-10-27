#pragma once
#if defined(__aarch64__) || defined(__arm__) || defined(__arm64__)

#include <cstddef> // For size_t
#include <cstdint> // For uint8_t
#include "Types.hpp"

// #define KAI_FP16_CAL

static int kai_thread_count = 4;

// --- 实现 1: float * qsi4c32 -> float---
size_t mllm_kleidai_get_packed_b_qsi4_size(int N, int K);
void mllm_kleidai_pack_b_and_bias_qsi4(uint8_t *packed_b_ptr, const float *b_ptr, const float *bias_ptr, int N, int K);
void mllm_kleidai_pack_b_and_bias_qsi4_quant(
    uint8_t *packed_b_ptr,
    const uint8_t *b_qweight_ptr,
    const float *b_scale_ptr,
    // const uint8_t *b_zero_ptr,
    const float *bias_ptr,
    int N,
    int K);
void mllm_kleidai_gemm_qsi4(float *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr, int M, int N, int K);

// --- 实现 1.5: float * qsi4c32 -> fp16---
size_t mllm_kleidai_get_packed_b_qsi4_size_to_fp16(int N, int K);
void mllm_kleidai_pack_b_and_bias_qsi4_to_fp16(uint8_t *packed_b_ptr, const float *b_ptr, const float *bias_ptr, int N, int K);
void mllm_kleidai_gemm_qsi4_to_fp16(mllm_fp16_t *c_ptr, const float *a_ptr, const uint8_t *packed_b_ptr, int M, int N, int K);

// --- 实现 2: float * fp16 -> float---

size_t mllm_kleidai_get_packed_b_fp16_size(int N, int K);
void mllm_kleidai_pack_b_and_bias_fp16(mllm_fp16_t *packed_b_ptr, const mllm_fp16_t *b_ptr, const float *bias_ptr, int N, int K);
void mllm_kleidai_gemm_fp16(float *c_ptr, const float *a_ptr, const mllm_fp16_t *packed_b_ptr, int M, int N, int K);

// --- 实现 3: float * fp32 -> float---
size_t mllm_kleidai_get_packed_b_fp32_size(int N, int K);
void mllm_kleidai_pack_b_and_bias_fp32(float *packed_b_ptr, const float *b_ptr, const float *bias_ptr, int N, int K);
void mllm_kleidai_gemm_fp32(float *c_ptr, const float *a_ptr, const float *packed_b_ptr, int M, int N, int K);

// --- APIs for Transposed Right-Hand Matrix Multiplication ---
void mllm_kleidai_gemm_fp32_transpose(float *c_ptr, const float *a_ptr, const float *b_ptr_nxk, const float *bias_ptr, int M, int N, int K);
void mllm_kleidai_gemm_fp16_transpose(float *c_ptr, const float *a_ptr, const mllm_fp16_t *b_ptr_nxk, const float *bias_ptr, int M, int N, int K);

// --- APIs for BSHD layout GEMM ---
void mllm_kleidai_gemm_fp32_bshd(float *c_ptr, const float *a_ptr, const float *packed_b_ptr, int B, int H, int S_M, int S_N, int D_K);
void mllm_kleidai_gemm_fp16_bshd(float *c_ptr, const float *a_ptr, const mllm_fp16_t *packed_b_ptr, int B, int H, int S_M, int S_N, int D_K);

#endif