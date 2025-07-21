/**
 * @file DataTypes.hpp
 * @author chenghua Wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-21
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include <half/half.hpp>

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
// Arm Device Has float16 native support
#include <arm_neon.h>
#endif

namespace mllm {

using mllm_fp64_t = double;
using mllm_fp32_t = float;
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
using mllm_fp16_t = float16_t;
#else
using mllm_fp16_t = half_float::half;
#endif
using mllm_int64_t = int64_t;
using mllm_uint64_t = uint64_t;
using mllm_int32_t = int32_t;
using mllm_uint32_t = uint32_t;
using mllm_in16_t = int16_t;
using mllm_uint16_t = uint16_t;
using mllm_int8_t = int8_t;
using mllm_uint8_t = uint8_t;

enum DataTypes {

};

}  // namespace mllm
