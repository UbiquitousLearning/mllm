#include <cmath>
#include "CPURMSNorm.hpp"
#include "Tensor.hpp"
#include "Timing.hpp"
#include "backends/cpu/third_party/ggml/VecDotFP32.hpp"
#include "backends/cpu/third_party/ggml/VecDotQ4.hpp"

namespace mllm {

// int32_t opp = 897988541;

// int32_t op_params[1];
CPURMSNorm::CPURMSNorm(Backend *bn, string opName, int normSize, float epsilon, bool add_unit_offset_, int threadCount) :
    thread_count(threadCount), add_unit_offset_(add_unit_offset_),
    Op(bn, opName), epsilon_(epsilon) {
    // op_params[0] = 897988541;s, sizeof(float));
    // memcpy(&epsilon_, op_param)
    normSize_ = normSize;
    weight_.setBackend(bn);
}

ErrorCode CPURMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // RMSNorm is similar to LayerNorm which operates on the channel dimension.
    assert(normSize_ == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    // outputs[0]->setDtype(activationDtype());
    // std::cout << name() << "  CPURMSNorm  reshape" << std::endl;
    return Op::reshape(inputs, outputs);
}

ErrorCode CPURMSNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    int batch = input->batch();
    int dim = input->dimension();
    int seq = input->sequence();
    int head = input->head();
#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; s++) {
                double sum_squares = 0.0F;
                // sum
                for (int d = 0; d < dim; d++) {
                    float value = input->dataAt<float>(n, h, s, d);
                    sum_squares += (double)value * value;
                }
                const float mean = sum_squares / dim;
                const float rms = 1.0f / sqrtf(mean + epsilon_);

                memcpy(outputs[0]->ptrAt<float>(n, h, s, 0),
                       inputs[0]->ptrAt<float>(n, h, s, 0),
                       dim * sizeof(float));
                vec_scale_f32(dim, outputs[0]->ptrAt<float>(n, h, s, 0), rms);
            }
        }
    }

// #pragma omp parallel for collapse(4) num_threads(thread_count)
//     for (int h = 0; h < head; h++) {
//         for (int n = 0; n < batch; n++) {
//             for (int s = 0; s < seq; s++) {
//                 for (int d = 0; d < dim; d++) {
//                     float weight = weight_.dataAt<float>(0, 0, 0, d);
//                     if (add_unit_offset_) {
//                         *outputs[0]->ptrAt<float>(n, h, s, d) *= (1 + weight);
//                     } else {
//                         *outputs[0]->ptrAt<float>(n, h, s, d) *= (weight);
//                     }
//                 }
//             }
//         }
//     }
// 第二部分：应用权重乘法
#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int h = 0; h < head; h++) {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; s++) {
                float *output_vec_ptr = outputs[0]->ptrAt<float>(n, h, s, 0);

                // 根据 weight_ 的数据类型处理
                if (weight_.dtype() == MLLM_TYPE_F32) {
                    // 如果权重是FP32类型
                    float *weight_vec_ptr = weight_.ptrAt<float>(0, 0, 0, 0);
                    if (add_unit_offset_) {
                        // 如果需要加1，创建 (1 + weight) 的临时向量
                        float *adjusted_weight_vec = new float[dim];
                        for (int d_idx = 0; d_idx < dim; ++d_idx) {
                            adjusted_weight_vec[d_idx] = 1.0f + weight_vec_ptr[d_idx];
                        }
                        // 进行逐元素乘法：output_vec_ptr *= adjusted_weight_vec
                        vec_mul_fp32(dim, output_vec_ptr, output_vec_ptr, adjusted_weight_vec);
                        delete[] adjusted_weight_vec; // 释放临时内存
                    } else {
                        // 直接使用 vec_mul_fp32 进行逐元素乘法：output_vec_ptr *= weight_vec_ptr
                        vec_mul_fp32(dim, output_vec_ptr, output_vec_ptr, weight_vec_ptr);
                    }
                } else if (weight_.dtype() == MLLM_TYPE_Q4_0) {
                    // 如果权重是 Q4_0 类型

                    if (add_unit_offset_) {
                        // 场景：output_fp32[i] = output_fp32_original[i] * (1.0f + Dequantize(weight_q4_0[i]))
                        // 这里的 `+1` 操作需要浮点数精度，因此必须先反量化 Q4_0 权重。
                        // 这将导致操作回到 FP32 * FP32 的逐元素乘法。
                        float *dequantized_and_adjusted_weight_fp32 = new float[dim];
                        // 反量化 Q4_0 权重
                        dequantize_row_q4_0(weight_.ptrAt<float>(0, 0, 0, 0), dequantized_and_adjusted_weight_fp32, dim);
                        // 逐元素添加 1
                        for (int i = 0; i < dim; ++i) {
                            dequantized_and_adjusted_weight_fp32[i] = 1.0f + dequantized_and_adjusted_weight_fp32[i];
                        }
                        // 执行 FP32 向量的逐元素乘法：output_vec_ptr *= dequantized_and_adjusted_weight_fp32
                        vec_mul_fp32(dim, output_vec_ptr, output_vec_ptr, dequantized_and_adjusted_weight_fp32);
                        delete[] dequantized_and_adjusted_weight_fp32; // 释放临时内存

                    } else {
                        // 场景：output_fp32[i] = Dequantize(output_q8_0[i]) * Dequantize(weight_q4_0[i])
                        // 这是用户期望的 Q4_0 * Q8_0 混合精度逐元素乘法。
                        // 首先，将当前 FP32 输出向量量化为临时的 Q8_0 缓冲区。
                        block_q8_0 *temp_output_q8_0 = new block_q8_0[dim / QK8_0];
                        quantize_row_q8_0(output_vec_ptr, temp_output_q8_0, dim);

                        // 执行 Q4_0 权重和 Q8_0 量化输出之间的逐元素乘法
                        // 结果直接存储回 output_vec_ptr (FP32)
                        vec_mul_q4_0_q8_0(dim, output_vec_ptr, weight_.hostPtr<void>(), temp_output_q8_0);

                        delete[] temp_output_q8_0; // 释放临时内存
                    }
                } else {
                    // 对于不支持的权重类型，此处断言以指示错误。
                    assert(false && "Unsupported weight_ dtype in CPURMSNorm::execute");
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}
ErrorCode CPURMSNorm::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, 1, normSize_); //
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        // auto l = loader.length(weight_.name());
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    return Op::load(loader);
}
ErrorCode CPURMSNorm::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm