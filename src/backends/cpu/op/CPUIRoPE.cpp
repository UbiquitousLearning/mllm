
#include "CPUIRoPE.hpp"
#include "Log.h"
#include "Types.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include "backends/cpu/quantize/QuantizeQ8.hpp"

namespace mllm {

vector<vector<int>> CPUIRoPE::sin_;
vector<vector<int>> CPUIRoPE::cos_;
int CPUIRoPE::sin_max;
int CPUIRoPE::cos_max;
int CPUIRoPE::global_pose_type_ = -1;
int CPUIRoPE::ishape_old;

void sinusoidal_position_embedding_llama(int seq_len, int output_dim, vector<vector<int>> &sin, vector<vector<int>> &cos, int &sin_max, int &cos_max) {
    vector<vector<float>> sin_fp(seq_len, vector<float>(output_dim));
    vector<vector<float>> cos_fp(seq_len, vector<float>(output_dim));
    sin.clear();
    cos.clear();
    sin.resize(seq_len, vector<int>(output_dim));
    cos.resize(seq_len, vector<int>(output_dim));

#pragma omp parallel for num_threads(4)
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim; d += 2) {
            int i = (int)d / 2;
            float sin_value = std::sin(s / std::pow(10000, 2.0 * i / output_dim));
            float cos_value = std::cos(s / std::pow(10000, 2.0 * i / output_dim));
            sin_fp[s][d] = sin_value;
            cos_fp[s][d] = cos_value;
            if (d + 1 < output_dim) {
                sin_fp[s][d + 1] = sin_value;
                cos_fp[s][d + 1] = cos_value;
            }
        }
    }
    // 计算 sin_fp 和 cos_fp 的最大绝对值
    sin_max = 0;
    cos_max = 0;
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim; ++d) {
            sin_max = std::max(sin_max, static_cast<int>(std::abs(sin_fp[s][d])));
            cos_max = std::max(cos_max, static_cast<int>(std::abs(cos_fp[s][d])));
        }
    }
    // 归一化并转换为 int8
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim; ++d) {
            sin[s][d] = static_cast<int>(std::round(sin_fp[s][d] / sin_max * 127));
            cos[s][d] = static_cast<int>(std::round(cos_fp[s][d] / cos_max * 127));
        }
    }
}
void sinusoidal_position_embedding_huggingface(int seq_len, int output_dim, vector<vector<int>> &sin, vector<vector<int>> &cos, int &sin_max, int &cos_max, int base = 10000) {
    vector<vector<float>> sin_fp(seq_len, vector<float>(output_dim));
    vector<vector<float>> cos_fp(seq_len, vector<float>(output_dim));
    sin.clear();
    cos.clear();
    sin.resize(seq_len, vector<int>(output_dim));
    cos.resize(seq_len, vector<int>(output_dim));
#pragma omp parallel for num_threads(4)
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim / 2; d += 1) {
            int i = (int)d / 1;
            float sin_value = sinf(s / std::pow(base, 2.0 * i / output_dim));
            float cos_value = cosf(s / std::pow(base, 2.0 * i / output_dim));
            sin_fp[s][d] = sin_value;
            cos_fp[s][d] = cos_value;
        }
        for (int d = output_dim / 2; d < output_dim; d += 1) {
            int i = (int)(d - output_dim / 2);
            float sin_value = sinf(s / std::pow(base, 2.0 * i / output_dim));
            float cos_value = cosf(s / std::pow(base, 2.0 * i / output_dim));
            sin_fp[s][d] = sin_value;
            cos_fp[s][d] = cos_value;
        }
    }
    sin_max = 0;
    cos_max = 0;
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim; ++d) {
            sin_max = std::max(sin_max, static_cast<int>(std::abs(sin_fp[s][d])));
            cos_max = std::max(cos_max, static_cast<int>(std::abs(cos_fp[s][d])));
        }
    }
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim; ++d) {
            sin[s][d] = static_cast<int>(std::round(sin_fp[s][d] / sin_max * 127));
            cos[s][d] = static_cast<int>(std::round(cos_fp[s][d] / cos_max * 127));
        }
    }
}

CPUIRoPE::CPUIRoPE(Backend *bn, string opName, int pose_type, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
}

CPUIRoPE::CPUIRoPE(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
    rope_theta_ = rope_theta;
    pos_max_ = max_position_embeddings;
}

CPUIRoPE::CPUIRoPE(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
    rope_theta_ = rope_theta;
    partial_rotary_factor_ = partial_rotary_factor;
    pos_max_ = max_position_embeddings;
}

ErrorCode CPUIRoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPUIRoPE  reshape" << std::endl;
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    ishape = inputs[0]->dimension() * partial_rotary_factor_;
    // pos_max_ = 16384;
    if (sin_.empty() || ishape_old < ishape || global_pose_type_ != pose_type_) {
        global_pose_type_ = pose_type_;
        ishape_old = ishape;
        if (pose_type_ == LLAMAROPE) {
            sinusoidal_position_embedding_llama(pos_max_, ishape, sin_, cos_, sin_max, cos_max);
        } else if (pose_type_ == PERSIMMONROPE) {
            sinusoidal_position_embedding_huggingface(pos_max_, ishape / 2, sin_, cos_, sin_max, cos_max, 25000);
        } else if (pose_type_ == HFHUBROPE || pose_type_ == MLAROPE) {
            sinusoidal_position_embedding_huggingface(pos_max_, ishape, sin_, cos_, sin_max, cos_max, rope_theta_);
        } else {
        }
    }
#ifdef USE_QNN
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
    if (cpuBackend->isStageSwitching()) {
        h_cnt_ = cpuBackend->getCurSequenceLength();
    }
#endif
    return Op::reshape(inputs, outputs);
}

void CPUIRoPE::rope_llama(shared_ptr<Tensor> input, shared_ptr<Tensor> output) {
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequance
                for (int d = 0; d < partial_dimension; d += 2) {
                    float in_value = input->dataAt<float>(n, h, s, d);
                    float in_value_2 = input->dataAt<float>(n, h, s, d + 1);
                    float sin_value = static_cast<float>(sin_[s + h_cnt_][d]) / 127 * sin_max;
                    float cos_value = static_cast<float>(cos_[s + h_cnt_][d]) / 127 * cos_max;
                    auto value = in_value * cos_value - in_value_2 * sin_value;
                    auto value2 = in_value * sin_value + in_value_2 * cos_value;
                    if (out_dtype == MLLM_TYPE_F32) {
                        output->setDataAt<float>(n, h, s, d, value);
                        output->setDataAt<float>(n, h, s, d + 1, value2);
                    } else if (out_dtype == MLLM_TYPE_F16) {
                        output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                        output->setDataAt<mllm_fp16_t>(n, h, s, d + 1, MLLM_FP32_TO_FP16(value2));
                    }
                }
            }
        }
    }
}
void CPUIRoPE::rope_hf(shared_ptr<Tensor> input, shared_ptr<Tensor> output) {
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
    int half = (int)(partial_dimension / 2);
    assert(partial_dimension % 2 == 0);
    if (output->ctype() == BSHD) {
        if (input->dtype() == MLLM_TYPE_F16) {
#pragma omp parallel for collapse(4) num_threads(thread_count)
            for (int n = 0; n < input->batch(); ++n) {
                for (int h = 0; h < input->head(); ++h) {
                    for (int s = 0; s < input->sequence(); ++s) { // sequance
                        for (int d = 0; d < partial_dimension / 2; ++d) {
                            auto v = input->ptrAt<mllm_fp16_t>(n, h, s, d);
                            auto o = output->ptrAt<mllm_fp16_t>(n, h, s, d);
                            float in_value = static_cast<float>(v[0]);
                            float in_value_2 = static_cast<float>(v[half]);
                            float sin_value = static_cast<float>(sin_[s + h_cnt_][d]) / 127 * sin_max;
                            auto c = static_cast<float>(cos_[s + h_cnt_][d]);
                            float cos_value = c / 127 * cos_max;
                            auto value = in_value * cos_value - in_value_2 * sin_value;
                            auto value2 = in_value * sin_value + in_value_2 * cos_value;
                            o[0] = MLLM_FP32_TO_FP16(value);
                            o[half] = MLLM_FP32_TO_FP16(value2);
                        }
                    }
                }
            }

        } else if (out_dtype == MLLM_TYPE_F32) {
#pragma omp parallel for collapse(4) num_threads(thread_count)
            for (int n = 0; n < input->batch(); ++n) {
                for (int h = 0; h < input->head(); ++h) {
                    for (int s = 0; s < input->sequence(); ++s) { // sequance
                        for (int d = 0; d < partial_dimension / 2; ++d) {
                            auto v = input->ptrAt<float>(n, h, s, d);
                            auto o = output->ptrAt<float>(n, h, s, d);
                            float in_value = v[0];
                            float in_value_2 = v[half];
                            float sin_value = static_cast<float>(sin_[s + h_cnt_][d]) / 127 * sin_max;
                            auto c = static_cast<float>(cos_[s + h_cnt_][d]);
                            float cos_value = c / 127 * cos_max;
                            auto value = in_value * cos_value - in_value_2 * sin_value;
                            auto value2 = in_value * sin_value + in_value_2 * cos_value;
                            o[0] = value;
                            o[half] = value2;
                        }
                    }
                }
            }
        } else if (out_dtype == MLLM_TYPE_F16) {
#pragma omp parallel for collapse(4) num_threads(thread_count)
            for (int n = 0; n < input->batch(); ++n) {
                for (int h = 0; h < input->head(); ++h) {
                    for (int s = 0; s < input->sequence(); ++s) { // sequance
                        for (int d = 0; d < partial_dimension / 2; ++d) {
                            auto v = input->ptrAt<float>(n, h, s, d);
                            auto o = output->ptrAt<mllm_fp16_t>(n, h, s, d);
                            float in_value = v[0];
                            float in_value_2 = v[half];
                            float sin_value = static_cast<float>(sin_[s + h_cnt_][d]) / 127 * sin_max;
                            float cos_value = static_cast<float>(cos_[s + h_cnt_][d]) / 127 * cos_max;
                            auto value = in_value * cos_value - in_value_2 * sin_value;
                            auto value2 = in_value * sin_value + in_value_2 * cos_value;
                            o[0] = MLLM_FP32_TO_FP16(value);
                            o[half] = MLLM_FP32_TO_FP16(value2);
                        }
                    }
                }
            }
        }
        return;
    }
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequance
                for (int d = 0; d < partial_dimension / 2; ++d) {
                    if (input->dtype() == MLLM_TYPE_F16) {
                        float in_value = static_cast<float>(input->dataAt<mllm_fp16_t>(n, h, s, d));
                        float in_value_2 = static_cast<float>(input->dataAt<mllm_fp16_t>(n, h, s, d + partial_dimension / 2));
                        float sin_value = sin_[s + h_cnt_][d];
                        float cos_value = cos_[s + h_cnt_][d];
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
                        float in_value = input->dataAt<float>(n, h, s, d);
                        float in_value_2 = input->dataAt<float>(n, h, s, d + partial_dimension / 2);
                        float sin_value = static_cast<float>(sin_[s + h_cnt_][d]) / 127 * sin_max;
                        float cos_value = static_cast<float>(cos_[s + h_cnt_][d]) / 127 * cos_max;
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
void CPUIRoPE::rope_permission(shared_ptr<Tensor> input, shared_ptr<Tensor> output) {
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequance
                for (int d = 0; d < partial_dimension; ++d) {
                    float in_value = input->dataAt<float>(n, h, s, d);
                    float in_value_2;
                    float sin_value = static_cast<float>(sin_[s + h_cnt_][d]) / 127 * sin_max;
                    float cos_value = static_cast<float>(cos_[s + h_cnt_][d]) / 127 * cos_max;
                    if (d < partial_dimension / 4) {
                        in_value_2 = -input->dataAt<float>(n, h, s, d + partial_dimension / 4);
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        if (out_dtype == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                        } else if (out_dtype == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                        }
                    } else if (d < (partial_dimension / 2)) {
                        in_value_2 = input->dataAt<float>(n, h, s, d - partial_dimension / 4);
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        if (out_dtype == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                        } else if (out_dtype == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                        }
                    } else {
                        if (out_dtype == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, in_value);
                        } else if (out_dtype == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(in_value));
                        }
                    }
                }
            }
        }
    }
}
void CPUIRoPE::rope_mla(shared_ptr<Tensor> input, shared_ptr<Tensor> output) {
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequance
                for (int d = 0; d < partial_dimension; ++d) {
                    int half_dim = input->dimension() / 2;
                    float in_value = input->dataAt<float>(n, h, s, d);
                    if (d < half_dim) {
                        in_value = input->dataAt<float>(n, h, s, d * 2);
                    } else {
                        in_value = input->dataAt<float>(n, h, s, 2 * (d - half_dim) + 1);
                    }
                    float in_value_2;
                    if (d < half_dim) {
                        in_value_2 = -input->dataAt<float>(n, h, s, 2 * d + 1);
                    } else {
                        in_value_2 = input->dataAt<float>(n, h, s, 2 * (d - half_dim));
                    }
                    // no change
                    float sin_value = static_cast<float>(sin_[s + h_cnt_][d]) / 127 * sin_max;
                    float cos_value = static_cast<float>(cos_[s + h_cnt_][d]) / 127 * cos_max;
                    auto value = in_value * cos_value + in_value_2 * sin_value;
                    if (out_dtype == MLLM_TYPE_F32) {
                        output->setDataAt<float>(n, h, s, d, value);
                    } else if (out_dtype == MLLM_TYPE_F16) {
                        output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                    }
                }
            }
        }
    }
}

// TODO: Q8_0 KVCache can not use!!
ErrorCode CPUIRoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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
#pragma omp parallel for collapse(3) num_threads(thread_count)
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
ErrorCode CPUIRoPE::doExecute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // if use QNN, when a new prompt input, the seq should be reset to 0 here as the setUp is not called
#ifdef USE_QNN
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
    if (cpuBackend->isStageSwitching() && cpuBackend->getExecutionType() == PROMPT) {
        h_cnt_ = 0;
    }
#endif

    auto &input = inputs[0];
    auto &output = outputs[0];
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
    // auto start_t = mllm_time_us();
    if (pose_type_ == LLAMAROPE) {
        rope_llama(input, output);
    } else if (pose_type_ == HFHUBROPE) {
        rope_hf(input, output);
    } else if (pose_type_ == PERSIMMONROPE) {
        rope_permission(input, output);
    } else if (pose_type_ == MLAROPE) {
        rope_mla(input, output);
    } else {
        MLLM_LOG_ERROR_STREAM << "RoPE type error" << std::endl;
    }
#pragma omp parallel for collapse(4) num_threads(thread_count)
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
    if (h_cnt_ >= pos_max_) {
        h_cnt_ = 0;
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUIRoPE::load(AbstructLoader &loader) {
    return Op::load(loader);
}
ErrorCode CPUIRoPE::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm
