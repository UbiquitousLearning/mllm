
#include "CPURoPE.hpp"
#include "Types.hpp"
#include <cmath>

namespace mllm {

vector<vector<float>> CPURoPE::sin_;
vector<vector<float>> CPURoPE::cos_;
int CPURoPE::global_pose_type_ = -1;
int CPURoPE::ishape_old;

void sinusoidal_position_embedding_llama(int seq_len, int output_dim, vector<vector<float>> &sin, vector<vector<float>> &cos) {
    sin.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        sin[i].resize(output_dim);
    }
    cos.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        cos[i].resize(output_dim);
    }
#pragma omp parallel for num_threads(4)
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim; d += 2) {
            int i = (int)d / 2;
            float sin_value = std::sin(s / std::pow(10000, 2.0 * i / output_dim));
            float cos_value = std::cos(s / std::pow(10000, 2.0 * i / output_dim));
            sin[s][d] = sin_value;
            cos[s][d] = cos_value;
            if (d + 1 < output_dim) {
                sin[s][d + 1] = sin_value;
                cos[s][d + 1] = cos_value;
            }
        }
    }
}
void sinusoidal_position_embedding_huggingface(int seq_len, int output_dim, vector<vector<float>> &sin, vector<vector<float>> &cos, int base = 10000) {
    sin.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        sin[i].resize(output_dim);
    }
    cos.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        cos[i].resize(output_dim);
    }
#pragma omp parallel for num_threads(4)
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim / 2; d += 1) {
            int i = (int)d / 1;
            float sin_value = sinf(s / std::pow(base, 2.0 * i / output_dim));
            float cos_value = cosf(s / std::pow(base, 2.0 * i / output_dim));
            sin[s][d] = sin_value;
            cos[s][d] = cos_value;
        }
        for (int d = output_dim / 2; d < output_dim; d += 1) {
            int i = (int)(d - output_dim / 2);
            float sin_value = sinf(s / std::pow(base, 2.0 * i / output_dim));
            float cos_value = cosf(s / std::pow(base, 2.0 * i / output_dim));
            sin[s][d] = sin_value;
            cos[s][d] = cos_value;
        }
    }
}

CPURoPE::CPURoPE(Backend *bn, string opName, int pose_type, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
}

CPURoPE::CPURoPE(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
    rope_theta_ = rope_theta;
    pos_max_ = max_position_embeddings;
}

CPURoPE::CPURoPE(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
    rope_theta_ = rope_theta;
    partial_rotary_factor_ = partial_rotary_factor;
    pos_max_ = max_position_embeddings;
}

ErrorCode CPURoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPURoPE  reshape" << std::endl;
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    ishape = inputs[0]->dimension() * partial_rotary_factor_;
    // pos_max_ = 16384;
    if (sin_.empty() || ishape_old < ishape || global_pose_type_ != pose_type_) {
        global_pose_type_ = pose_type_;
        ishape_old = ishape;
        if (pose_type_ == LLAMAROPE) {
            sinusoidal_position_embedding_llama(pos_max_, ishape, sin_, cos_);
        } else if (pose_type_ == PERSIMMONROPE) {
            sinusoidal_position_embedding_huggingface(pos_max_, ishape / 2, sin_, cos_, 25000);
        } else if (pose_type_ == HFHUBROPE || pose_type_ == MLAROPE) {
            sinusoidal_position_embedding_huggingface(pos_max_, ishape, sin_, cos_, rope_theta_);
        } else {
        }
    }
#ifdef USE_QNN
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
    if (cpuBackend->isStageSwitching()) {
        h_cnt_ = cpuBackend->getSequenceLength();
    }
#endif
    return Op::reshape(inputs, outputs);
}

ErrorCode CPURoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &input = inputs[0];
    auto &output = outputs[0];
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequance
#pragma omp parallel for num_threads(thread_count)
                for (int d = 0; d < partial_dimension; ++d) {
                    if (pose_type_ == LLAMAROPE) {
                        float in_value = input->dataAt<float>(n, h, s, d);
                        float in_value_2;
                        if (d % 2 == 0) { // if is even number: 0,2,4
                            in_value_2 = -input->dataAt<float>(n, h, s, d + 1);
                        } else {
                            in_value_2 = input->dataAt<float>(n, h, s, d - 1);
                        }
                        float sin_value = sin_[s + h_cnt_][d];
                        float cos_value = cos_[s + h_cnt_][d];
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        if (out_dtype == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                        } else if (out_dtype == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                        }
                    } else if (pose_type_ == PERSIMMONROPE) {
                        float in_value = input->dataAt<float>(n, h, s, d);
                        float in_value_2;
                        float sin_value = sin_[s + h_cnt_][d];
                        float cos_value = cos_[s + h_cnt_][d];
                        if (d < partial_dimension / 4) {
                            in_value_2 = -input->dataAt<float>(n, h, s, d + partial_dimension / 4);
                            auto value = in_value * cos_value + in_value_2 * sin_value;
                            if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F32) {
                                output->setDataAt<float>(n, h, s, d, value);
                            } else if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F16) {
                                output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                            }
                        } else if (d < (partial_dimension / 2)) {
                            in_value_2 = input->dataAt<float>(n, h, s, d - partial_dimension / 4);
                            auto value = in_value * cos_value + in_value_2 * sin_value;
                            if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F32) {
                                output->setDataAt<float>(n, h, s, d, value);
                            } else if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F16) {
                                output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                            }
                        } else {
                            if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F32) {
                                output->setDataAt<float>(n, h, s, d, in_value);
                            } else if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F16) {
                                output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(in_value));
                            }
                        }
                    } else if (pose_type_ == HFHUBROPE) {
                        float in_value = input->dataAt<float>(n, h, s, d);
                        float in_value_2;
                        if (d < (partial_dimension / 2)) {
                            in_value_2 = -input->dataAt<float>(n, h, s, d + partial_dimension / 2);
                        } else {
                            in_value_2 = input->dataAt<float>(n, h, s, d - partial_dimension / 2);
                        }
                        float sin_value = sin_[s + h_cnt_][d];
                        float cos_value = cos_[s + h_cnt_][d];
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                        } else if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                        }
                    } else if (pose_type_ == MLAROPE) {
                        int half_dim = input->dimension() / 2;
                        float in_value = input->dataAt<float>(n, h, s, d);
                        if (d < half_dim) {
                            in_value = input->dataAt<float>(n, h, s, d * 2);
                        } else {
                            in_value = input->dataAt<float>(n, h, s, 2 *(d - half_dim)+1);
                        }
                        float in_value_2;
                        if (d < half_dim) {
                            in_value_2 = -input->dataAt<float>(n, h, s, 2 *d+1);
                        } else {
                            in_value_2 = input->dataAt<float>(n, h, s, 2 *(d - half_dim));
                        }
                        // no change
                        float sin_value = sin_[s + h_cnt_][d];
                        float cos_value = cos_[s + h_cnt_][d];
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                        } else if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                        }
                    } else {
                        std::cerr << "RoPE type error" << std::endl;
                    }
                }
            }
        }
    }
    h_cnt_ += input->sequence();
    if (h_cnt_ > pos_max_) {
        h_cnt_ = 0;
    }

    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) {
#pragma omp parallel for num_threads(thread_count)
                for (int d = partial_dimension; d < input->dimension(); ++d) {
                    if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F32) {
                        output->setDataAt<float>(n, h, s, d, input->dataAt<float>(n, h, s, d));
                    } else if (output->dtypeAt(n, h, s, d) == MLLM_TYPE_F16) {
                        output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(input->dataAt<float>(n, h, s, d)));
                    }
                }
            }
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPURoPE::load(AbstructLoader &loader) {
    return Op::load(loader);
}
ErrorCode CPURoPE::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm
