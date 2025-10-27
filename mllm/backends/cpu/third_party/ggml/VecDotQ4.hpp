/*
 * This code is based on mllm(https://github.com/ggerganov/mllm),
 * please see https://github.com/ggerganov/mllm/blob/master/src/mllm.c
 * mllm is licensed under MIT Copyright (c) 2022 Georgi Gerganov:
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
#include "DataType.hpp"
#include "QuantizeFP16.hpp"
#include <cassert>
#include <cstddef>

inline void vec_mul_q4_0_q8_0(const int n, float *__restrict result_fp32, const void *__restrict q4_operand_ptr, const void *__restrict q8_operand_ptr) {
    // QK4_0 和 QK8_0 在 ggml 中通常是 32，表示每个量化块的元素数量。
    // 确保维度 n 是块大小的倍数。
    assert(n % QK4_0 == 0); // Q4_0 block size
    assert(n % QK8_0 == 0); // Q8_0 block size

    // 计算总的块数量
    const int num_blocks = n / QK4_0; // 假设 QK4_0 == QK8_0 或它们是兼容的块大小

    const block_q4_0 *q4_blocks = static_cast<const block_q4_0 *>(q4_operand_ptr);
    const block_q8_0 *q8_blocks = static_cast<const block_q8_0 *>(q8_operand_ptr);

    int current_fp32_idx = 0; // 用于跟踪 FP32 结果数组中的当前位置

    for (int i = 0; i < num_blocks; ++i) {
        // 获取当前 Q4_0 和 Q8_0 块的尺度因子
        const float d_q4 = MLLM_FP16_TO_FP32(q4_blocks[i].d);
        const float d_q8 = MLLM_FP16_TO_FP32(q8_blocks[i].d);
        const float combined_scale = d_q4 * d_q8; // 组合尺度

        // 获取当前块的量化数据指针
        const uint8_t *qs_q4 = q4_blocks[i].qs;
        const int8_t *qs_q8 = q8_blocks[i].qs;

        // 逐元素处理块内的量化数据
        // QK4_0 / 2 是因为 Q4_0 的每个 uint8_t 存储了两个 4-bit 元素（nibbles）
        for (int j = 0; j < QK4_0 / 2; ++j) {
            // 处理第一个 4-bit 元素
            // (value - 8) 是 Q4_0 的反量化零点调整
            const int q4_val_low_nibble = (qs_q4[j] & 0x0F) - 8;
            // Q8_0 直接使用 8-bit 值
            const int q8_val_first_byte = qs_q8[j * 2];

            // 逐元素相乘并乘以组合尺度，结果存入 FP32 数组
            result_fp32[current_fp32_idx + j] = (float)q4_val_low_nibble * (float)q8_val_first_byte * combined_scale;

            // 处理第二个 4-bit 元素
            const int q4_val_high_nibble = (qs_q4[j] >> 4) - 8;
            const int q8_val_second_byte = qs_q8[j * 2 + 1];

            result_fp32[current_fp32_idx + j + QK4_0 / 2] = (float)q4_val_high_nibble * (float)q8_val_second_byte * combined_scale;
        }
        // 更新 FP32 结果数组的索引，移动到下一个块的起始位置
        current_fp32_idx += QK4_0;
    }
}

void vec_dot_q4_K_q8_K(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy);
void vec_dot_q4_0_q8_0(const int n, float *__restrict s, const void *__restrict vx, const void *__restrict vy);