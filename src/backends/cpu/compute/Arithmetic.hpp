#ifndef MLLM_ARITHMETIC_HPP
#define MLLM_ARITHMETIC_HPP
#include "Tensor.hpp"
#include "Types.hpp"
#include <functional>
#include "ParamLoader.hpp"


#include <chrono>
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#endif

using namespace mllm;

void mllm_add_fp32(float* a, float*b, float*c, int n);
void mllm_sub_fp32(float* a, float* b, float* c, int n);
void mllm_mul_fp32(float* a, float* b, float* c, int n);
void mllm_div_fp32(float* a, float* b, float* c, int n);

void mllm_add_fp32(float* a, float value, float* c, int n);
void mllm_sub_fp32(float* a, float value, float* c, int n);
void mllm_mul_fp32(float* a, float value, float* c, int n);
void mllm_div_fp32(float* a, float value, float* c, int n);
#endif //MLLM_ARITHMETIC_HPP