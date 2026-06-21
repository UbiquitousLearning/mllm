// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <cmath>

namespace mllm::cpu {

/**
 * @brief GDN 纯计算 Kernel
 * 对应公式: S_t = alpha * S_{t-1} - alpha * beta * (S_{t-1} * k) * k^T + beta * v * k^T
 */
struct GDNKernel {
    static constexpr int D = 128; // Head Dim (对应 Qwen3.5-0.8B 配置)
    static constexpr int H = 16;  // Heads

    template <typename T>
    static void forward(
        const T* s_prev, const T* k, const T* v, 
        const T* gate, const T* beta,
        T* s_new, 
        float* output = nullptr, const float* q = nullptr, // 👈 改成 float*
        int batch_size = 1
    ) {
        for (int b = 0; b < batch_size; ++b) {
            int offset_h = b * H;
            int offset_s = b * H * D * D;
            int offset_vec = b * H * D;

            for (int h = 0; h < H; ++h) {
                const T* s_ptr = s_prev + offset_s + h * D * D;
                const T* k_ptr = k + offset_vec + h * D;
                const T* v_ptr = v + offset_vec + h * D;
                T* s_out_ptr = s_new + offset_s + h * D * D;
                
                // 计算 alpha = exp(gate)
                T alpha_val = std::exp(gate[offset_h + h]); 
                T beta_val = beta[offset_h + h];

                // 1. 全局衰减: S_temp = alpha * S_prev
                for (int i = 0; i < D * D; ++i) {
                    s_out_ptr[i] = alpha_val * s_ptr[i];
                }

                // 2. 擦除项: erase = beta * (S_temp * k) * k^T
                std::vector<T> proj(D, 0);
                for (int i = 0; i < D; ++i) {
                    T sum = 0;
                    for (int j = 0; j < D; ++j) {
                        sum += s_out_ptr[i * D + j] * k_ptr[j];
                    }
                    proj[i] = sum;
                }
                for (int i = 0; i < D; ++i) {
                    T factor = beta_val * proj[i];
                    for (int j = 0; j < D; ++j) {
                        s_out_ptr[i * D + j] -= factor * k_ptr[j];
                    }
                }

                // 3. 写入项: write = beta * v * k^T
                for (int i = 0; i < D; ++i) {
                    T factor = beta_val * v_ptr[i];
                    for (int j = 0; j < D; ++j) {
                        s_out_ptr[i * D + j] += factor * k_ptr[j];
                    }
                }
                
                // 4. 可选：计算输出 o = S_new * q
                if (q && output) {
                    const T* q_ptr = q + offset_vec + h * D;
                    T* o_ptr = output + offset_vec + h * D;
                    for (int i = 0; i < D; ++i) {
                        T sum = 0;
                        for (int j = 0; j < D; ++j) {
                            sum += s_out_ptr[i * D + j] * q_ptr[j];
                        }
                        o_ptr[i] = sum;
                    }
                }
            }
        }
    }
};

} // namespace mllm::cpu
