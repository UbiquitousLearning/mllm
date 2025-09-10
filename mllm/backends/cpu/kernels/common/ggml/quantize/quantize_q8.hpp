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

#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

void quantize_row_q8_0(const float* __restrict x, void* __restrict y, int k);
void dequantize_row_q8_0(const void* __restrict vx, float* __restrict y, int k);

void quantize_row_q8_K(const float* __restrict x, void* __restrict y, int k);
void dequantize_row_q8_K(const block_q8_K* __restrict x, float* __restrict y, int k);

// for per-tensor int8 quantize
void quantize_row_i8(const float* __restrict x, void* __restrict y, int k, float scale = 1.f);
void dequantize_row_i8(const void* __restrict vx, float* __restrict y, int k, float scale = 1.f);
void dequantize_row_i8_to_fp16(const void* __restrict vx, void* __restrict vy, int k, float scale = 1.f);
void quantize_round_dequantize_row_i8(const float* __restrict vx, float* __restrict y, int k, float scale = 1.f);

}  // namespace mllm::cpu
