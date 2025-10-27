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

#ifndef MLLM_TYPE_HPP
#define MLLM_TYPE_HPP

#define MLLM_RESTRICT __restrict
#include "Types.hpp"

typedef void (*mllm_to_float_func)(const void *src, float *dst, const int n); // from src type to float(stored in dst)  n is the number of element in src
typedef void (*mllm_from_float_func)(const float *src, void *dst, const int n);
typedef void (*mllm_vec_dot_func)(const int n, float *MLLM_RESTRICT dst, const void *MLLM_RESTRICT x, const void *MLLM_RESTRICT y);
typedef void (*mllm_from_float_to_mat_func)(const float *MLLM_RESTRICT x, void *MLLM_RESTRICT y, int64_t nr, int64_t k, int64_t bs);
typedef void (*mllm_vec_add_row_func)(const int n, const void *MLLM_RESTRICT src, float *MLLM_RESTRICT dst, const float alpha);
typedef void (*gemv_func)(int n, float *MLLM_RESTRICT s, size_t bs, const void *MLLM_RESTRICT x,
                          const void *MLLM_RESTRICT y, int nr, int nc, const void *MLLM_RESTRICT bias);
typedef void (*gemm_func)(int n, float *MLLM_RESTRICT s, size_t bs, const void *MLLM_RESTRICT x,
                          const void *MLLM_RESTRICT y, int nr, int nc, const void *MLLM_RESTRICT bias);

typedef struct type_traits_t {
    size_t size;   // type size
    int blck_size; // number of element in a block (quantization block)
    int blck_size_interleave;
    mllm_to_float_func to_float;
    mllm_from_float_func from_float;
    mllm_from_float_to_mat_func from_float_to_mat;
    mllm_vec_dot_func vec_dot;
    DataType vec_dot_type;            // vec_dot do dot product between two DataType, this is the other type
    mllm_vec_add_row_func add_row_to; // add alpha * row to a row of float
    gemv_func gemv;
    gemm_func gemm;
} type_traits_t;

extern type_traits_t type_traits[];

inline size_t type_size(DataType type) {
    return type_traits[type].size;
}

inline int blck_size(DataType type) {
    return type_traits[type].blck_size;
}

// ne: number of elements in a row
// return the number of bytes in a row
inline size_t row_size(DataType type, int64_t ne) {
    return type_size(type) * ne / blck_size(type);
}

#endif // MLLM_TYPE_HPP
