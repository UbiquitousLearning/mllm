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

#pragma once

#include <cstddef>
#include "mllm/core/DataTypes.hpp"

#define MLLM_RESTRICT __restrict

namespace mllm::cpu::ggml {

// function pointer types
using mllm_to_float_func = void (*)(const void*, float*,
                                    const int);  // from src type to float(stored in dst)  n is the number of element in src
using mllm_from_float_func = void (*)(const float*, void*, const int);
using mllm_vec_dot_func = void (*)(const int, float*, const void*, const void*);
using mllm_from_float_to_mat_func = void (*)(const float*, void*, int64_t, int64_t, int64_t);
using mllm_vec_add_row_func = void (*)(const int, const void*, float*, const float);
using mllm_gemv_func = void (*)(int, float*, size_t, const void*, const void*, int, int, const void*);
using mllm_gemm_func = void (*)(int, float*, size_t, const void*, const void*, int, int, const void*);

template<DataTypes DT>
struct TypeTraits {
  static const size_t size = 0;
  static const int blck_size = 1;
  static const int blck_size_interleave = 1;

  static const mllm_to_float_func to_float;
  static const mllm_from_float_func from_float;
  static const mllm_from_float_to_mat_func from_float_to_mat;
  static const mllm_vec_dot_func vec_dot;
  static const DataTypes vec_dot_type;
  static const mllm_vec_add_row_func add_row_to;
  static const mllm_gemv_func gemv;
  static const mllm_gemm_func gemm;
};

template<DataTypes DT>
const mllm_to_float_func TypeTraits<DT>::to_float = nullptr;
template<DataTypes DT>
const mllm_from_float_func TypeTraits<DT>::from_float = nullptr;
template<DataTypes DT>
const mllm_from_float_to_mat_func TypeTraits<DT>::from_float_to_mat = nullptr;
template<DataTypes DT>
const mllm_vec_dot_func TypeTraits<DT>::vec_dot = nullptr;
template<DataTypes DT>
const DataTypes TypeTraits<DT>::vec_dot_type = DT;
template<DataTypes DT>
const mllm_vec_add_row_func TypeTraits<DT>::add_row_to = nullptr;
template<DataTypes DT>
const mllm_gemv_func TypeTraits<DT>::gemv = nullptr;
template<DataTypes DT>
const mllm_gemm_func TypeTraits<DT>::gemm = nullptr;

#define DECLARE_TYPE_TRAITS(DT)                                 \
  template<>                                                    \
  struct TypeTraits<DT> {                                       \
    static const size_t size;                                   \
    static const int blck_size;                                 \
    static const int blck_size_interleave;                      \
    static const mllm_to_float_func to_float;                   \
    static const mllm_from_float_func from_float;               \
    static const mllm_from_float_to_mat_func from_float_to_mat; \
    static const mllm_vec_dot_func vec_dot;                     \
    static const DataTypes vec_dot_type;                        \
    static const mllm_vec_add_row_func add_row_to;              \
    static const mllm_gemv_func gemv;                           \
    static const mllm_gemm_func gemm;                           \
  };

DECLARE_TYPE_TRAITS(MLLM_TYPE_F32)
DECLARE_TYPE_TRAITS(MLLM_TYPE_F16)
DECLARE_TYPE_TRAITS(MLLM_TYPE_Q4_0)
DECLARE_TYPE_TRAITS(MLLM_TYPE_Q4_K)
DECLARE_TYPE_TRAITS(MLLM_TYPE_Q6_K)
DECLARE_TYPE_TRAITS(MLLM_TYPE_Q8_0)
DECLARE_TYPE_TRAITS(MLLM_TYPE_Q8_K)
DECLARE_TYPE_TRAITS(MLLM_TYPE_Q2_K)
DECLARE_TYPE_TRAITS(MLLM_TYPE_Q3_K)
DECLARE_TYPE_TRAITS(MLLM_TYPE_IQ2_XXS)

// helper functions
template<DataTypes DT>
inline size_t type_size() {
  return TypeTraits<DT>::size;
}

template<DataTypes DT>
inline int blck_size() {
  return TypeTraits<DT>::blck_size;
}

// ne: number of elements in a row
// return the number of bytes in a row
template<DataTypes DT>
inline size_t row_size(int64_t ne) {
  return type_size<DT>() * ne / blck_size<DT>();
}

}  // namespace mllm::cpu::ggml
