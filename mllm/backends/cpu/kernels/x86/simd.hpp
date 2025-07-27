/**
 * @file simd.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-27
 *
 */
#pragma once

#include "mllm/utils/CPUArchHelper.hpp"

#if defined(MLLM_HOST_FEATURE_AVX512F)
#include <immintrin.h>
#elif defined(MLLM_HOST_FEATURE_AVX2) || defined(MLLM_HOST_FEATURE_AVX)
#include <immintrin.h>
#elif defined(MLLM_HOST_FEATURE_SSE2)
#include <emmintrin.h>
#elif defined(MLLM_HOST_FEATURE_SSE)
#include <xmmintrin.h>
#endif
