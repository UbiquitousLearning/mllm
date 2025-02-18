
#include "CPUMultimodalRoPE.hpp"
// #include "Timing.hpp"
#include "Types.hpp"
#include <cassert>
#include <cmath>
// #include <iostream>
#include "backends/cpu/quantize/QuantizeQ8.hpp"

namespace mllm {

vector<float> CPUMultimodalRoPE::theta_; //inv_freq

vector<vector<float>> CPUMultimodalRoPE::sin_;
vector<vector<float>> CPUMultimodalRoPE::cos_;
int CPUMultimodalRoPE::ishape_old;

typedef float (*mllm_rope_init_func)(const OpParam &, std::vector<float>&);

float multimodal_default_init_rope(const OpParam& config, vector<float>& theta) {
    auto base = config.at("base");  // theta_i = base^-(2i/dim) = 1 / base^(2i/dim)    i from 0 to (dim/2 - 1)
    auto dim = config.at("dim");

    theta.resize((int)(dim/2));
#pragma omp parallel for num_threads(4)
    for (int i = 0;i < theta.size();i++)
        theta[i] = 1.0 / pow(base, 2.0 * i / dim);

    return  1.0;
}

void apply_multimodal_rotary_pos_emb(
    std::vector<std::vector<float>>& cos,
    std::vector<std::vector<float>>& sin,
    const std::vector<int>& mrope_section,
    int unsqueeze_dim = 1
) {
    // Assuming all vectors in 'cos' and 'sin' have the same size for simplicity
    size_t original_size = cos[0].size();
    std::vector<std::vector<float>> new_cos(cos.size(), std::vector<float>(original_size, 0));
    std::vector<std::vector<float>> new_sin(sin.size(), std::vector<float>(original_size, 0));
    size_t start = 0;
    for (size_t i = 0; i < mrope_section.size(); ++i) {
        int section_size = mrope_section[i];
        for (size_t row = 0; row < cos.size(); ++row) {
            for (int col = 0; col < section_size; ++col) {
                // Calculate target index by considering modulo operation for rotation
                size_t target_index = col + (i % 3) * section_size;
                if(target_index >= new_cos[row].size()) {
                    // Adjust target index to fit within the bounds of the vector
                    target_index %= new_cos[row].size();
                }
                new_cos[row][target_index] = cos[row][start + col];
                new_sin[row][target_index] = sin[row][start + col];
            }
        }
        start += section_size;
    }
    // Apply changes back to the original vectors
    cos = new_cos;
    sin = new_sin;
}

void multimodal_sinusoidal_position_embedding(int seq_len, int output_dim, const vector<float>& theta,
                                               vector<vector<float>> &sin, vector<vector<float>> &cos, float attention_scaling = 1.0,
                                               const std::vector<int>& mrope_section = {}) {
    sin.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        sin[i].resize(output_dim);
    }
    cos.resize(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        cos[i].resize(output_dim);
    }

    auto mid = output_dim / 2;

#pragma omp parallel for num_threads(4)
    for (int s = 0; s < seq_len; ++s) {
        for (int d = 0; d < output_dim / 2; d += 1) {
            int i = d;
            auto t = s * theta[i];
            float sin_value = sinf(t);
            float cos_value = cosf(t);
            sin[s][d] = sin_value;
            cos[s][d] = cos_value;
            if (d + mid < output_dim) {
                sin[s][d + mid] = sin_value * attention_scaling;
                cos[s][d + mid] = cos_value * attention_scaling;
            }
        }
    }

    //ADD HERE TODO
    if(!mrope_section.empty()){
        apply_multimodal_rotary_pos_emb(cos, sin, mrope_section);
    }
}

CPUMultimodalRoPE::CPUMultimodalRoPE(Backend *bn, string opName, float rope_theta, int max_position_embeddings, vector<int> mrope_section, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    rope_theta_ = rope_theta;
    pos_max_ = max_position_embeddings;
    mrope_section_ = mrope_section;
    for (int i = 0; i < mrope_section.size(); i++) {
        mrope_section_.push_back(mrope_section[i]);
    }
}

ErrorCode CPUMultimodalRoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
//    std::cout << name() << "  CPUMultimodalRoPE  reshape" << std::endl;
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    ishape = inputs[0]->dimension() * partial_rotary_factor_;
    // pos_max_ = 16384;

    if (sin_.empty() || ishape_old < ishape ) {
        auto config = config_;
        config["base"] = (float)rope_theta_;
        config["dim"] = ishape;
        float attention_scaling = multimodal_default_init_rope(config, theta_);
        ishape_old = ishape;
        multimodal_sinusoidal_position_embedding(pos_max_, ishape, theta_, sin_, cos_, attention_scaling, mrope_section_);
    }
#ifdef USE_QNN
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
    if (cpuBackend->isStageSwitching()) {
        h_cnt_ = cpuBackend->getCurSequenceLength();
    }
#endif
    return Op::reshape(inputs, outputs);
}


void CPUMultimodalRoPE::multimodal_rope_hf(shared_ptr<Tensor> input, shared_ptr<Tensor> output) {
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
                            float sin_value = sin_[s + h_cnt_][d];
                            float cos_value = cos_[s + h_cnt_][d];
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
#pragma omp parallel for collapse(4) num_threads(thread_count)
                for (int n = 0; n < input->batch(); ++n) {
                    for (int h = 0; h < input->head(); ++h) {
                        for (int s = 0; s < input->sequence(); ++s) { // sequance
                            for (int d = 0; d < partial_dimension / 2; ++d) {
                                auto v = input->ptrAt<float>(n, h, s, d);
                                auto o = output->ptrAt<float>(n, h, s, d);
                                float in_value = v[0];
                                float in_value_2 = v[half];
                                float sin_value = sin_[s + h_cnt_][d];
                                float cos_value = cos_[s + h_cnt_][d];
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
                                float sin_value = sin_[s + h_cnt_][d];
                                float cos_value = cos_[s + h_cnt_][d];
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
                    }
                }
            }
        }
    }
}

// TODO: Q8_0 KVCache can not use!!
ErrorCode CPUMultimodalRoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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
ErrorCode CPUMultimodalRoPE::doExecute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &input = inputs[0];
    auto &output = outputs[0];
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
    // auto start_t = mllm_time_us();
    multimodal_rope_hf(input, output);
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

ErrorCode CPUMultimodalRoPE::load(AbstructLoader &loader) {
    return Op::load(loader);
}
ErrorCode CPUMultimodalRoPE::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm
