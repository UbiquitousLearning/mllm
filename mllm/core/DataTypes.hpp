// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <complex>
#include <half/half.hpp>

#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
// Arm Device Has float16 native support
#include <arm_neon.h>
#endif

namespace mllm {

//===----------------------------------------------------------------------===//
// C & C++ Compiler Builtin Types Define
//===----------------------------------------------------------------------===//
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
using mllm_int16_t = int16_t;
using mllm_uint16_t = uint16_t;
using mllm_int8_t = int8_t;
using mllm_uint8_t = uint8_t;
using mllm_byte_t = mllm_uint8_t;

using mllm_complex_fp32_t = std::complex<mllm_fp32_t>;
using mllm_complex_fp64_t = std::complex<mllm_fp64_t>;

//===----------------------------------------------------------------------===//
// GGUF Types Define
//===----------------------------------------------------------------------===//
/*
 * This code is based on ggml(https://github.com/ggerganov/ggml),
 * please see https://github.com/ggerganov/ggml/blob/master/src/ggml.c
 * ggml is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
 *
 * MIT License
 * Copyright (c) 2022 Georgi Gerganov
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// #define MLLM_QKK_64
#ifdef MLLM_QKK_64
/**
 * @def QK_K
 * @brief The number of elements in a quantization block. Set to 64.
 */
#define QK_K 64

/**
 * @def K_SCALE_SIZE
 * @brief The size of the scale data in bytes for QK_K blocks. Set to 4.
 */
#define K_SCALE_SIZE 4
#else
/**
 * @def QK_K
 * @brief The number of elements in a quantization block. Set to 256.
 */
#define QK_K 256

/**
 * @def K_SCALE_SIZE
 * @brief The size of the scale data in bytes for QK_K blocks. Set to 12.
 */
#define K_SCALE_SIZE 12
#endif

/**
 * @def QK4_0
 * @brief The number of elements in a Q4_0 quantization block.
 */
#define QK4_0 32

/**
 * @def QK8_0
 * @brief The number of elements in a Q8_0 quantization block.
 */
#define QK8_0 32

#pragma pack(push, 1)

/**
 * @struct block_q4_0
 * @brief 4-bit quantization block.
 *
 * Each block contains 32 4-bit quantized values.
 */
typedef struct {         // NOLINT
  mllm_fp16_t d;         /**< The delta (scaling factor) for this block. */
  uint8_t qs[QK4_0 / 2]; /**< The 32 quantized values, packed as 4-bit nibbles. */
} block_q4_0;
using mllm_block_q4_0_t = block_q4_0;

/**
 * @struct block_q4_K
 * @brief 4-bit quantization block with K-level quantization.
 * @note This is a more complex 4-bit quantization scheme with super-blocks.
 * The weight is represented as `x = a * q + b`.
 * Effectively uses 4.5 bits per weight.
 */
#ifdef MLLM_QKK_64
typedef struct {        // NOLINT
  mllm_fp16_t d[2];     /**< Super-block scales/mins. */
  uint8_t scales[2];    /**< 4-bit block scales/mins. */
  uint8_t qs[QK_K / 2]; /**< 4-bit quantized values. */
} block_q4_K;
using mllm_block_q4_K_t = block_q4_K;
static_assert(sizeof(block_q4_K) == 2 * sizeof(uint16_t) + QK_K / 2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {                // NOLINT
  mllm_fp16_t d;                /**< Super-block scale for quantized scales. */
  mllm_fp16_t dmin;             /**< Super-block scale for quantized mins. */
  uint8_t scales[K_SCALE_SIZE]; /**< Scales and mins, quantized with 6 bits. */
  uint8_t qs[QK_K / 2];         /**< 4-bit quantized values. */
} block_q4_K;
using mllm_block_q4_K_t = block_q4_K;
static_assert(sizeof(block_q4_K) == 2 * sizeof(mllm_fp16_t) + K_SCALE_SIZE + QK_K / 2, "wrong q4_K block size/padding");
#endif

/**
 * @struct block_q6_K
 * @brief 6-bit quantization block with K-level quantization.
 */
typedef struct {            // NOLINT
  uint8_t ql[QK_K / 2];     /**< Lower 4 bits of the quantized values. */
  uint8_t qh[QK_K / 4];     /**< Upper 2 bits of the quantized values. */
  int8_t scales[QK_K / 16]; /**< Scales, quantized with 8 bits. */
  mllm_fp16_t d;            /**< Super-block scale. */
} block_q6_K;
using mllm_block_q6_K_t = block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(mllm_fp16_t) + QK_K / 16 + 3 * QK_K / 4, "wrong q6_K block size/padding");

/**
 * @struct block_q8_0
 * @brief 8-bit quantization block.
 */
typedef struct {    // NOLINT
  mllm_fp16_t d;    /**< The delta (scaling factor) for this block. */
  int8_t qs[QK8_0]; /**< The 32 quantized values as 8-bit signed integers. */
} block_q8_0;
using mllm_block_q8_0_t = block_q8_0;

/**
 * @struct block_q8_per_tensor
 * @brief Per-tensor 8-bit quantization block.
 * @note Used in `vecdot_i8_i8`. TODO: remove this structure if possible.
 */
typedef struct {    // NOLINT
  int8_t qs[QK8_0]; /**< The 32 quantized values as 8-bit signed integers. */
} block_q8_per_tensor;
using mllm_block_q8_per_tensor_t = block_q8_per_tensor;

/**
 * @struct block_q8_K
 * @brief 8-bit quantization block with K-level quantization.
 * @note This is only used for intermediate quantization and dot products.
 */
typedef struct {            // NOLINT
  float d;                  /**< The delta (scaling factor) for this block as a 32-bit float. */
  int8_t qs[QK_K];          /**< The quantized values as 8-bit signed integers. */
  int16_t bsums[QK_K / 16]; /**< Sum of quants in groups of 16, used for speeding up calculations. */
} block_q8_K;
using mllm_block_q8_K_t = block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K / 16 * sizeof(int16_t), "wrong q8_K block size/padding");

/**
 * @struct block_q4_0x4
 * @brief A group of four `block_q4_0` blocks.
 */
typedef struct {         // NOLINT
  mllm_fp16_t d[4];      /**< Deltas for 4 q4_0 blocks. */
  uint8_t qs[QK4_0 * 2]; /**< Quantized values for 4 q4_0 blocks. */
} block_q4_0x4;
using mllm_block_q4_0x4_t = block_q4_0x4;
static_assert(sizeof(block_q4_0x4) == 4 * sizeof(mllm_fp16_t) + QK4_0 * 2, "wrong q4_0x4 block size/padding");

/**
 * @struct block_q4_0x8
 * @brief A group of eight `block_q4_0` blocks.
 */
typedef struct {         // NOLINT
  mllm_fp16_t d[8];      /**< Deltas for 8 q4_0 blocks. */
  uint8_t qs[QK4_0 * 4]; /**< Quantized values for 8 q4_0 blocks. */
} block_q4_0x8;
using mllm_block_q4_0x8_t = block_q4_0x8;
static_assert(sizeof(block_q4_0x8) == 8 * sizeof(mllm_fp16_t) + QK4_0 * 4, "wrong q4_0x8 block size/padding");

/**
 * @struct block_q8_0x4
 * @brief A group of four `block_q8_0` blocks.
 */
typedef struct {        // NOLINT
  mllm_fp16_t d[4];     /**< Deltas for 4 q8_0 blocks. */
  int8_t qs[QK8_0 * 4]; /**< Quantized values for 4 q8_0 blocks. */
} block_q8_0x4;
using mllm_block_q8_0x4_t = block_q8_0x4;
static_assert(sizeof(block_q8_0x4) == 4 * sizeof(mllm_fp16_t) + QK8_0 * 4, "wrong q8_0x4 block size/padding");

/**
 * @struct block_q8_0x8
 * @brief A group of eight `block_q8_0` blocks.
 */
typedef struct {        // NOLINT
  mllm_fp16_t d[8];     /**< Deltas for 8 q8_0 blocks. */
  int8_t qs[QK8_0 * 8]; /**< Quantized values for 8 q8_0 blocks. */
} block_q8_0x8;
using mllm_block_q8_0x8_t = block_q8_0x8;
static_assert(sizeof(block_q8_0x8) == 8 * sizeof(mllm_fp16_t) + QK8_0 * 8, "wrong q8_0x8 block size/padding");

/**
 * @struct block_q2_K
 * @brief 2-bit quantization block with K-level quantization.
 */
typedef struct {             // NOLINT
  uint8_t scales[QK_K / 16]; /**< Scales and mins, quantized with 4 bits. */
  uint8_t qs[QK_K / 4];      /**< Quantized values (2 bits per value). */
  mllm_fp16_t d;             /**< Super-block scale for quantized scales. */
  mllm_fp16_t dmin;          /**< Super-block scale for quantized mins. */
} block_q2_K;
using mllm_block_q2_K_t = block_q2_K;
static_assert(sizeof(block_q2_K) == 2 * sizeof(mllm_fp16_t) + QK_K / 16 + QK_K / 4, "wrong q2_K block size/padding");

/**
 * @struct block_q3_K
 * @brief 3-bit quantization block with K-level quantization.
 */
typedef struct {           // NOLINT
  uint8_t hmask[QK_K / 8]; /**< High bit of the quantized values. */
  uint8_t qs[QK_K / 4];    /**< Low 2 bits of the quantized values. */
  uint8_t scales[12];      /**< Scales, quantized with 6 bits. */
  mllm_fp16_t d;           /**< Super-block scale. */
} block_q3_K;
using mllm_block_q3_K_t = block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(mllm_fp16_t) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

/**
 * @struct block_iq2_xxs
 * @brief 2-bit "imbalanced" quantization block.
 */
typedef struct {         // NOLINT
  mllm_fp16_t d;         /**< The delta (scaling factor) for this block. */
  uint16_t qs[QK_K / 8]; /**< Quantized values. */
} block_iq2_xxs;
using mllm_block_iq2_xxs_t = block_iq2_xxs;
static_assert(sizeof(block_iq2_xxs) == sizeof(mllm_fp16_t) + QK_K / 8 * sizeof(uint16_t), "wrong iq2_xxs block size/padding");

//===----------------------------------------------------------------------===//
// MXFP4
//
// https://arxiv.org/pdf/2310.10537
//
// 4 bits (1 sign, 2 exponent, 1 mantissa) plus 1 shared power-of-two scale per 32 value block
//===----------------------------------------------------------------------===//
typedef struct {     // NOLINT
  uint8_t data[16];  // 32 x fp4, 128 bits, 16 bytes
  uint8_t scale;
} mllm_mxfp4_t;
static_assert(sizeof(mllm_mxfp4_t) == 17, "wrong mxfp4 size/padding");

#pragma pack(pop)

//===----------------------------------------------------------------------===//
// MLLM Types Enum
//
// The ID of each datatype is highly correlated with the mllm model file.
//
// If you want to add new datatype, it's restrict to add at the tail of this enum.
//===----------------------------------------------------------------------===//
enum DataTypes : int32_t {
  // MLLM V1 Datatypes.
  kFloat32 = 0,
  kFloat16 = 1,
  kGGUF_Q4_0 = 2,
  kGGUF_Q4_1 = 3,
  kGGUF_Q8_0 = 8,
  kGGUF_Q8_1 = 9,
  kGGUF_Q8_Pertensor = 10,
  kGGUF_Q4_K = 12,
  kGGUF_Q6_K = 14,
  kGGUF_Q8_K = 15,
  kInt8 = 16,
  kInt16 = 17,
  kInt32 = 18,
  kGGUF_Q4_0_4_4 = 19,
  kGGUF_Q4_0_4_8 = 20,
  kGGUF_Q4_0_8_8 = 21,
  kGGUF_Q8_0_4_4 = 22,
  kGGUF_Q3_K = 23,
  kGGUF_Q2_K = 24,
  kGGUF_Q1_K = 25,
  kGGUF_IQ2_XXS = 26,
  kGGUF_IQ2_XS = 27,
  kGGUF_IQ1_S = 28,
  kGGUF_IQ1_M = 29,
  kGGUF_IQ2_S = 30,

  // 31-127 is left for other GGUF Datatypes.

  // MLLM V2 Datatypes.
  kBFloat16 = 128,
  kUInt8 = 129,
  kUInt16 = 130,
  kUInt32 = 131,
  kInt64 = 132,
  kUInt64 = 133,

  // Byte is used to store general data. Such as
  // 1. Special packed kleidiai weight and bias
  // 2. Your customized quantization method.
  kByte = 134,
  kMXFP4 = 135,

  // complex dtypes for STFT and other ops
  kComplexFloat32 = 201,
  kComplexFloat64 = 202,
};

//===----------------------------------------------------------------------===//
// TypeInfo based on C++ types
//===----------------------------------------------------------------------===//
template<typename T>
struct DataTypeInfo {
  using dtype_t = T;

  static inline dtype_t zero() {
    static_assert(sizeof(T) == 0, "DataTypeInfo::zero not specialized for this type");
    return dtype_t();
  };

  static inline dtype_t one() {
    static_assert(sizeof(T) == 0, "DataTypeInfo::one not specialized for this type");
    return dtype_t();
  };

  static inline dtype_t max() {
    static_assert(sizeof(T) == 0, "DataTypeInfo::max not specialized for this type");
    return dtype_t();
  };

  static inline dtype_t min() {
    static_assert(sizeof(T) == 0, "DataTypeInfo::min not specialized for this type");
    return dtype_t();
  };

  /**
   * @brief How many elements are in a type.
   *
   * @return size_t
   */
  static inline size_t lanes() { return 1; };

  /**
   * @briefT The size of this type.
   *
   * @return size_t
   */
  static inline size_t bytes() { return sizeof(dtype_t); };

  static inline std::string name() { return "Unknown"; };
};

#define MLLM_DEFINE_BASIC_TYPE_INFO(T, zero_val, one_val, max_val, min_val, str_name) \
  template<>                                                                          \
  struct DataTypeInfo<T> {                                                            \
    using dtype_t = T;                                                                \
    static inline dtype_t zero() { return zero_val; }                                 \
    static inline dtype_t one() { return one_val; }                                   \
    static inline dtype_t max() { return max_val; }                                   \
    static inline dtype_t min() { return min_val; }                                   \
    static inline size_t lanes() { return 1; }                                        \
    static inline size_t bytes() { return sizeof(dtype_t); }                          \
    static inline std::string name() { return str_name; }                             \
  }

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_fp64_t, 0.0, 1.0, std::numeric_limits<mllm_fp64_t>::max(),
                            std::numeric_limits<mllm_fp64_t>::lowest(), "Float64");

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_fp32_t, 0.0f, 1.0f, std::numeric_limits<mllm_fp32_t>::max(),
                            std::numeric_limits<mllm_fp32_t>::lowest(), "Float32");

template<>
struct DataTypeInfo<mllm_fp16_t> {
  using dtype_t = mllm_fp16_t;
  static inline dtype_t zero() {
    if constexpr (std::is_same_v<mllm_fp16_t, half_float::half>) {
      return half_float::half(0.f);
    } else {
      // Machine support fp16.
      return static_cast<mllm_fp16_t>(0.f);
    }
  }

  static inline dtype_t one() {
    if constexpr (std::is_same_v<mllm_fp16_t, half_float::half>) {
      return half_float::half(1.f);
    } else {
      // Machine support fp16.
      return static_cast<mllm_fp16_t>(1.f);
    }
  }
  static inline dtype_t max() {
    if constexpr (std::is_same_v<mllm_fp16_t, half_float::half>) {
      return half_float::half(65504.f);
    } else {
      // Machine support fp16.
      return static_cast<mllm_fp16_t>(65504.f);
    }
  }
  static inline dtype_t min() {
    if constexpr (std::is_same_v<mllm_fp16_t, half_float::half>) {
      return half_float::half(-65504.f);
    } else {
      // Machine support fp16.
      return static_cast<mllm_fp16_t>(-65504.f);
    }
  }
  static inline size_t lanes() { return 1; }
  static inline size_t bytes() { return sizeof(dtype_t); }

  static inline std::string name() { return "Float16"; }
};

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_int64_t, 0, 1, std::numeric_limits<mllm_int64_t>::max(),
                            std::numeric_limits<mllm_int64_t>::min(), "Int64");

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_uint64_t, 0, 1, std::numeric_limits<mllm_uint64_t>::max(),
                            std::numeric_limits<mllm_uint64_t>::min(), "UInt64");

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_int32_t, 0, 1, std::numeric_limits<mllm_int32_t>::max(),
                            std::numeric_limits<mllm_int32_t>::min(), "Int32");

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_uint32_t, 0, 1, std::numeric_limits<mllm_uint32_t>::max(),
                            std::numeric_limits<mllm_uint32_t>::min(), "UInt32");

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_int16_t, 0, 1, std::numeric_limits<mllm_int16_t>::max(),
                            std::numeric_limits<mllm_int16_t>::min(), "Int16");

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_uint16_t, 0, 1, std::numeric_limits<mllm_uint16_t>::max(),
                            std::numeric_limits<mllm_uint16_t>::min(), "UInt16");

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_int8_t, 0, 1, std::numeric_limits<mllm_int8_t>::max(), std::numeric_limits<mllm_int8_t>::min(),
                            "Int8");

MLLM_DEFINE_BASIC_TYPE_INFO(mllm_uint8_t, 0, 1, std::numeric_limits<mllm_uint8_t>::max(),
                            std::numeric_limits<mllm_uint8_t>::min(), "UInt8");

// There is no need to declare mllm_byte_t. It's already declared in mllm_uint8_t.

// Complex types can not be declared by MLLM_DEFINE_BASIC_TYPE_INFO macro
template<>
struct DataTypeInfo<mllm_complex_fp64_t> {
  static inline mllm_complex_fp64_t zero() { return std::complex<mllm_fp64_t>{0.0, 0.0}; }
  static inline mllm_complex_fp64_t one() { return std::complex<mllm_fp64_t>{1.0, 0.0}; }
  static inline mllm_complex_fp64_t max() {
    return std::complex<mllm_fp64_t>{std::numeric_limits<mllm_fp64_t>::max(), std::numeric_limits<mllm_fp64_t>::max()};
  }
  static inline mllm_complex_fp64_t min() {
    return std::complex<mllm_fp64_t>{std::numeric_limits<mllm_fp64_t>::lowest(), std::numeric_limits<mllm_fp64_t>::lowest()};
  }
  static inline size_t lanes() { return 1; }
  static inline size_t bytes() { return sizeof(mllm_complex_fp64_t); }
  static inline std::string name() { return "ComplexFP64"; }
};

template<>
struct DataTypeInfo<mllm_complex_fp32_t> {
  static inline mllm_complex_fp32_t zero() { return std::complex<mllm_fp32_t>{0.0f, 0.0f}; }
  static inline mllm_complex_fp32_t one() { return std::complex<mllm_fp32_t>{1.0f, 0.0f}; }
  static inline mllm_complex_fp32_t max() {
    return std::complex<mllm_fp32_t>{std::numeric_limits<mllm_fp32_t>::max(), std::numeric_limits<mllm_fp32_t>::max()};
  }
  static inline mllm_complex_fp32_t min() {
    return std::complex<mllm_fp32_t>{std::numeric_limits<mllm_fp32_t>::lowest(), std::numeric_limits<mllm_fp32_t>::lowest()};
  }
  static inline size_t lanes() { return 1; }
  static inline size_t bytes() { return sizeof(mllm_complex_fp32_t); }
  static inline std::string name() { return "ComplexFP32"; }
};

#undef MLLM_DEFINE_BASIC_TYPE_INFO

#define MLLM_DEFINE_QUANT_TYPE_INFO(T, element_count, str_name) \
  template<>                                                    \
  struct DataTypeInfo<T> {                                      \
    using dtype_t = T;                                          \
    static inline dtype_t zero() { return dtype_t(); }          \
    static inline dtype_t one() { return dtype_t(); }           \
    static inline dtype_t max() { return dtype_t(); }           \
    static inline dtype_t min() { return dtype_t(); }           \
    static inline size_t lanes() { return element_count; }      \
    static inline size_t bytes() { return sizeof(dtype_t); }    \
    static inline std::string name() { return str_name; }       \
  }

MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q4_0_t, QK4_0, "GGUF_Q4_0");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q4_K_t, QK_K, "GGUF_Q4_K");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q6_K_t, QK_K, "GGUF_Q6_K");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q8_0_t, QK8_0, "GGUF_Q8_0");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q8_per_tensor_t, QK8_0, "GGUF_Q8_Per_Tensor");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q8_K_t, QK_K, "GGUF_Q8_K");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q4_0x4_t, QK4_0 * 4, "GGUF_Q4_0x4");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q4_0x8_t, QK4_0 * 8, "GGUF_Q4_0x8");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q8_0x4_t, QK8_0 * 4, "GGUF_Q8_0x4");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q8_0x8_t, QK8_0 * 8, "GGUF_Q8_0x8");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q2_K_t, QK_K, "GGUF_Q2_K");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_q3_K_t, QK_K, "GGUF_Q3_K");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_iq2_xxs_t, QK_K, "GGUF_IQ2_XXS");
MLLM_DEFINE_QUANT_TYPE_INFO(mllm_mxfp4_t, 32, "MXFP4");

#undef MLLM_DEFINE_QUANT_TYPE_INFO

//===----------------------------------------------------------------------===//
// TypeInfo based on MLLM enum types
//===----------------------------------------------------------------------===//
template<DataTypes __MLLM_TYPE>
struct MllmDataTypeInfo {};

#define MLLM_DEFINE_SELF_TYPE_INFO(__mllm_enum_type, __compiler_type)          \
  template<>                                                                   \
  struct MllmDataTypeInfo<__mllm_enum_type> {                                  \
    using dtype_t = __compiler_type;                                           \
    static inline dtype_t zero() { return DataTypeInfo<dtype_t>::zero(); }     \
    static inline dtype_t one() { return DataTypeInfo<dtype_t>::one(); }       \
    static inline dtype_t min() { return DataTypeInfo<dtype_t>::min(); }       \
    static inline dtype_t max() { return DataTypeInfo<dtype_t>::max(); }       \
    static inline size_t lanes() { return DataTypeInfo<dtype_t>::lanes(); }    \
    static inline size_t bytes() { return DataTypeInfo<dtype_t>::bytes(); }    \
    static inline std::string name() { return DataTypeInfo<dtype_t>::name(); } \
  }

MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kFloat32, mllm_fp32_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kFloat16, mllm_fp16_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kInt8, mllm_int8_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kUInt8, mllm_uint8_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kInt16, mllm_int16_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kUInt16, mllm_uint16_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kInt32, mllm_int32_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kUInt32, mllm_uint32_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kInt64, mllm_int64_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kUInt64, mllm_uint64_t);

MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q4_0, mllm_block_q4_0_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q4_K, mllm_block_q4_K_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q6_K, mllm_block_q6_K_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q8_0, mllm_block_q8_0_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q8_Pertensor, mllm_block_q8_per_tensor_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q8_K, mllm_block_q8_K_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q4_0_4_4, mllm_block_q4_0x4_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q4_0_4_8, mllm_block_q4_0x8_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q8_0_4_4, mllm_block_q8_0x4_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q2_K, mllm_block_q2_K_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_Q3_K, mllm_block_q3_K_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kGGUF_IQ2_XXS, mllm_block_iq2_xxs_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kMXFP4, mllm_mxfp4_t);

MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kComplexFloat64, mllm_complex_fp64_t);
MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kComplexFloat32, mllm_complex_fp32_t);

#undef MLLM_DEFINE_SELF_TYPE_INFO

size_t lanesOfType(DataTypes dtype);

size_t bytesOfType(DataTypes dtype);

std::string nameOfType(DataTypes dtype);

}  // namespace mllm

// WARN: The Macros below should not be used anymore.
//
// We left it here just for compatibility. It need to be removed one day.
#define MLLM_TYPE_F32 ::mllm::DataTypes::kFloat32
#define MLLM_TYPE_F16 ::mllm::DataTypes::kFloat16
#define MLLM_TYPE_Q4_0 ::mllm::DataTypes::kGGUF_Q4_0
#define MLLM_TYPE_Q4_1 ::mllm::DataTypes::kGGUF_Q4_1
#define MLLM_TYPE_Q8_0 ::mllm::DataTypes::kGGUF_Q8_0
#define MLLM_TYPE_Q8_1 ::mllm::DataTypes::kGGUF_Q8_1
#define MLLM_TYPE_Q8_PER_TENSOR ::mllm::DataTypes::kGGUF_Q8_Pertensor
#define MLLM_TYPE_Q4_K ::mllm::DataTypes::kGGUF_Q4_K
#define MLLM_TYPE_Q6_K ::mllm::DataTypes::kGGUF_Q6_K
#define MLLM_TYPE_Q8_K ::mllm::DataTypes::kGGUF_Q8_K
#define MLLM_TYPE_I8 ::mllm::DataTypes::kInt8
#define MLLM_TYPE_I16 ::mllm::DataTypes::kInt16
#define MLLM_TYPE_I32 ::mllm::DataTypes::kInt32
#define MLLM_TYPE_Q4_0_4_4 ::mllm::DataTypes::kGGUF_Q4_0_4_4
#define MLLM_TYPE_Q4_0_4_8 ::mllm::DataTypes::kGGUF_Q4_0_4_8
#define MLLM_TYPE_Q4_0_8_8 ::mllm::DataTypes::kGGUF_Q4_0_8_8
#define MLLM_TYPE_Q8_0_4_4 ::mllm::DataTypes::kGGUF_Q8_0_4_4
#define MLLM_TYPE_Q3_K ::mllm::DataTypes::kGGUF_Q3_K
#define MLLM_TYPE_Q2_K ::mllm::DataTypes::kGGUF_Q2_K
#define MLLM_TYPE_Q1_K ::mllm::DataTypes::kGGUF_Q1_K
#define MLLM_TYPE_IQ2_XXS ::mllm::DataTypes::kGGUF_IQ2_XXS
#define MLLM_TYPE_IQ2_XS ::mllm::DataTypes::kGGUF_IQ2_XS
#define MLLM_TYPE_IQ1_S ::mllm::DataTypes::kGGUF_IQ1_S
#define MLLM_TYPE_IQ1_M ::mllm::DataTypes::kGGUF_IQ1_M
#define MLLM_TYPE_IQ2_S ::mllm::DataTypes::kGGUF_IQ2_S
#define MLLM_TYPE_BFLOAT16 ::mllm::DataTypes::kBFloat16
#define MLLM_TYPE_UINT8 ::mllm::DataTypes::kUInt8
#define MLLM_TYPE_UINT16 ::mllm::DataTypes::kUInt16
#define MLLM_TYPE_UINT32 ::mllm::DataTypes::kUInt32
#define MLLM_TYPE_BYTE ::mllm::DataTypes::kByte
#define MLLM_TYPE_MXFP4 ::mllm::DataTypes::kMXFP4
