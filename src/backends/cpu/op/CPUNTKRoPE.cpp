/**
 * @file CPUNTKRoPE.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#include "CPUNTKRoPE.hpp"
#include "Types.hpp"
#include <cassert>
#include <cmath>
#include "backends/cpu/quantize/QuantizeQ8.hpp"

namespace mllm {

int CPUNTKRoPE::in_shape_old = 0;
std::vector<std::vector<float>> CPUNTKRoPE::emb_sin_;
std::vector<std::vector<float>> CPUNTKRoPE::emb_cos_;

namespace {
void get_sin_cos_emb_hf(
    std::vector<std::vector<float>> &emb_sin,
    std::vector<std::vector<float>> &emb_cos,
    int seq_len,
    int output_dim,
    float theta,
    std::vector<float> &long_factor,
    std::vector<float> &short_factor,
    int original_max_position_embeddings,
    int max_position_embeddings = 2048) {
    auto scale = (float)max_position_embeddings / (float)original_max_position_embeddings;
    auto scaling_factor = (float)std::sqrt(1 + std::log(scale) / std::log(original_max_position_embeddings));

    // compute sin and cos
    emb_sin.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        emb_sin[i].resize(output_dim);
    }
    emb_cos.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        emb_cos[i].resize(output_dim);
    }

    // get ext_factor
    std::vector<float> ext_factors;
    if (seq_len > original_max_position_embeddings)
        ext_factors = long_factor;
    else
        ext_factors = short_factor;

    // calculate inv_freq
    std::vector<float> inv_freq(output_dim / 2, 0.f);
    for (int i = 0; i < output_dim / 2; ++i) {
        inv_freq[i] = 1.f / (float)(std::pow(theta, (float)i / (float)output_dim));
    }

    std::vector<float> t(seq_len, 0.f);
    for (int s = 0; s < seq_len; ++s) t[s] = (float)s;

    std::vector<std::vector<float>> freqs;
    {
        int seq_len = t.size();
        int output_dim = inv_freq.size() * 2; // Since inv_freq is half the size of the final output dimension

        for (int i = 0; i < seq_len; ++i) {
            freqs.emplace_back(output_dim / 2, 0.f);
            for (int j = 0; j < output_dim / 2; ++j) {
                freqs[i][j] = t[i] * (1.0f / ext_factors[j]) * inv_freq[j];
            }
        }
    }

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < output_dim / 2; ++j) {
            emb_sin[i][j] = std::sin(freqs[i][j]) * scaling_factor;
            emb_cos[i][j] = std::cos(freqs[i][j]) * scaling_factor;
        }
        for (int j = output_dim / 2; j < output_dim; ++j) {
            emb_sin[i][j] = std::sin(freqs[i][j - output_dim / 2]) * scaling_factor;
            emb_cos[i][j] = std::cos(freqs[i][j - output_dim / 2]) * scaling_factor;
        }
    }
}

void apply_rope_hf(
    std::shared_ptr<Tensor> &input,
    std::shared_ptr<Tensor> &output,
    std::vector<std::vector<float>> &emb_sin,
    std::vector<std::vector<float>> &emb_cos,
    int h_cnt) {
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * 1;
    int half = (int)(partial_dimension / 2);
    assert(partial_dimension % 2 == 0);
    if (output->ctype() == BSHD) {
        if (input->dtype() == MLLM_TYPE_F16) {
#pragma omp parallel for collapse(4) num_threads(4)
            for (int n = 0; n < input->batch(); ++n) {
                for (int h = 0; h < input->head(); ++h) {
                    for (int s = 0; s < input->sequence(); ++s) { // sequence
                        for (int d = 0; d < partial_dimension / 2; ++d) {
                            auto v = input->ptrAt<mllm_fp16_t>(n, h, s, d);
                            auto o = output->ptrAt<mllm_fp16_t>(n, h, s, d);
                            float in_value = static_cast<float>(v[0]);
                            float in_value_2 = static_cast<float>(v[half]);
                            float sin_value = emb_sin[s + h_cnt][d];
                            float cos_value = emb_cos[s + h_cnt][d];
                            auto value = in_value * cos_value - in_value_2 * sin_value;
                            auto value2 = in_value * sin_value + in_value_2 * cos_value;
                            o[0] = MLLM_FP32_TO_FP16(value);
                            o[half] = MLLM_FP32_TO_FP16(value2);
                        }
                    }
                }
            }

        } else {
            if (out_dtype == MLLM_TYPE_F32) {
#pragma omp parallel for collapse(4) num_threads(4)
                for (int n = 0; n < input->batch(); ++n) {
                    for (int h = 0; h < input->head(); ++h) {
                        for (int s = 0; s < input->sequence(); ++s) { // sequence
                            for (int d = 0; d < partial_dimension / 2; ++d) {
                                auto v = input->ptrAt<float>(n, h, s, d);
                                auto o = output->ptrAt<float>(n, h, s, d);
                                float in_value = v[0];
                                float in_value_2 = v[half];
                                float sin_value = emb_sin[s + h_cnt][d];
                                float cos_value = emb_cos[s + h_cnt][d];
                                auto value = in_value * cos_value - in_value_2 * sin_value;
                                auto value2 = in_value * sin_value + in_value_2 * cos_value;
                                o[0] = value;
                                o[half] = value2;
                            }
                        }
                    }
                }
            } else if (out_dtype == MLLM_TYPE_F16) {
#pragma omp parallel for collapse(4) num_threads(4)
                for (int n = 0; n < input->batch(); ++n) {
                    for (int h = 0; h < input->head(); ++h) {
                        for (int s = 0; s < input->sequence(); ++s) { // sequence
                            for (int d = 0; d < partial_dimension / 2; ++d) {
                                auto v = input->ptrAt<float>(n, h, s, d);
                                auto o = output->ptrAt<mllm_fp16_t>(n, h, s, d);
                                float in_value = v[0];
                                float in_value_2 = v[half];
                                float sin_value = emb_sin[s + h_cnt][d];
                                float cos_value = emb_cos[s + h_cnt][d];
                                auto value = in_value * cos_value - in_value_2 * sin_value;
                                auto value2 = in_value * sin_value + in_value_2 * cos_value;
                                o[0] = MLLM_FP32_TO_FP16(value);
                                o[half] = MLLM_FP32_TO_FP16(value2);
                            }
                        }
                    }
                }
            }
        }
        return;
    }
#pragma omp parallel for collapse(4) num_threads(4)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequence
                for (int d = 0; d < partial_dimension / 2; ++d) {
                    if (input->dtype() == MLLM_TYPE_F16) {
                        float in_value = static_cast<float>(input->dataAt<mllm_fp16_t>(n, h, s, d));
                        float in_value_2 = static_cast<float>(input->dataAt<mllm_fp16_t>(n, h, s, d + partial_dimension / 2));
                        float sin_value = emb_sin[s + h_cnt][d];
                        float cos_value = emb_cos[s + h_cnt][d];
                        auto value = in_value * cos_value - in_value_2 * sin_value;
                        auto value2 = in_value * sin_value + in_value_2 * cos_value;
                        if (out_dtype == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                            output->setDataAt<float>(n, h, s, d + partial_dimension / 2, value2);
                        } else if (out_dtype == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                            output->setDataAt<mllm_fp16_t>(n, h, s, d + partial_dimension / 2, MLLM_FP32_TO_FP16(value2));
                        }

                    } else {
                        auto in_value = input->dataAt<float>(n, h, s, d);
                        auto in_value_2 = input->dataAt<float>(n, h, s, d + partial_dimension / 2);
                        float sin_value = emb_sin[s + h_cnt][d];
                        float cos_value = emb_cos[s + h_cnt][d];
                        auto value = in_value * cos_value - in_value_2 * sin_value;
                        auto value2 = in_value * sin_value + in_value_2 * cos_value;
                        if (out_dtype == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                            output->setDataAt<float>(n, h, s, d + partial_dimension / 2, value2);
                        } else if (out_dtype == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                            output->setDataAt<mllm_fp16_t>(n, h, s, d + partial_dimension / 2, MLLM_FP32_TO_FP16(value2));
                        }
                    }
                }
            }
        }
    }
}
} // namespace

CPUNTKRoPE::CPUNTKRoPE(Backend *bn, string op_name, int pose_type, int thread_count) :
    Op(bn, op_name), thread_count_(thread_count), pose_type_(pose_type) {
}

CPUNTKRoPE::CPUNTKRoPE(Backend *bn, string op_name, int pose_type, float rope_theta,
                       const std::vector<float> &long_factor,
                       const std::vector<float> &short_factor,
                       int original_max_position_embeddings,
                       int max_position_embeddings,
                       int thread_count) :
    Op(bn, op_name),
    thread_count_(thread_count),
    pose_type_(pose_type),
    rope_theta_(rope_theta),
    long_factor_(long_factor),
    short_factor_(short_factor),
    original_max_position_embeddings_(original_max_position_embeddings),
    max_position_embeddings_(max_position_embeddings) {
}

ErrorCode CPUNTKRoPE::doExecute(std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs) {
    auto &input = inputs[0];
    auto &output = outputs[0];
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * 1;
    switch ((RoPEType)pose_type_) {
    case RoPEType::HFHUBROPE:
        apply_rope_hf(input, output, emb_sin_, emb_cos_, h_cnt_);
        break;
    default:
        MLLM_LOG_ERROR("RoPEType={} is not supported yet. Currently, only support HFHUBROPE style NTKRoPE", pose_type_);
        break;
    }

#pragma omp parallel for collapse(4) num_threads(4)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) {
                for (int d = partial_dimension; d < input->dimension(); ++d) {
                    if (out_dtype == MLLM_TYPE_F32) {
                        output->setDataAt<float>(n, h, s, d, input->dataAt<float>(n, h, s, d));
                    } else if (out_dtype == MLLM_TYPE_F16) {
                        output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(input->dataAt<float>(n, h, s, d)));
                    }
                }
            }
        }
    }

    h_cnt_ += input->sequence();
    if (h_cnt_ >= max_position_embeddings_) {
        h_cnt_ = 0;
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUNTKRoPE::reshape(std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    in_shape = inputs[0]->dimension();
    if (emb_sin_.empty() || in_shape_old < in_shape) {
        in_shape_old = in_shape;
        switch ((RoPEType)pose_type_) {
        case RoPEType::HFHUBROPE:
            get_sin_cos_emb_hf(
                emb_sin_,
                emb_cos_,
                max_position_embeddings_,
                inputs[0]->dimension(),
                rope_theta_,
                long_factor_,
                short_factor_,
                original_max_position_embeddings_,
                max_position_embeddings_);
            break;
        default:
            MLLM_LOG_ERROR("RoPEType={} is not supported yet. Currently, only support HFHUBROPE style NTKRoPE", pose_type_);
            break;
        }
    }
    return Op::reshape(inputs, outputs);
    return MLLM_NO_ERROR;
}

ErrorCode CPUNTKRoPE::execute(std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs) {
    if (outputs[0]->dtype() == MLLM_TYPE_Q8_0) {
        auto tmp_out = std::make_shared<Tensor>(outputs[0]->backend());
        // tmp_out->setBackend(outputs[0]->backend());
        auto b = outputs[0]->batch();
        auto h = outputs[0]->head();
        auto d = outputs[0]->dimension();
        auto s = outputs[0]->sequence();
        tmp_out->chls() = outputs[0]->chls();
        tmp_out->setCtype(outputs[0]->ctype());
        tmp_out->reshape(b, h, s, d);
        tmp_out->setDtype(MLLM_TYPE_F32);
        tmp_out->alloc();
        doExecute(inputs, {tmp_out});
#pragma omp parallel for collapse(3) num_threads(4)
        for (int b = 0; b < tmp_out->batch(); b++) {
            for (int h = 0; h < tmp_out->head(); h++) {
                for (int s = 0; s < tmp_out->sequence(); s++) {
                    quantize_row_q8_0(tmp_out->hostPtr<float>() + tmp_out->offset(b, h, s, 0),
                                      (char *)outputs[0]->rawHostPtr()
                                          + outputs[0]->offset(b, h, s, 0) * sizeof(block_q8_0) / QK8_0,
                                      tmp_out->dimension());
                }
            }
        }
        return MLLM_NO_ERROR;
    } else {
        return doExecute(inputs, outputs);
    }
}

ErrorCode CPUNTKRoPE::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUNTKRoPE::free(std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm