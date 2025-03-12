//
// Created by Rongjie Yi on 25-2-16.
//

#ifndef CPUAPPLYVISIONROPEFUNC_HPP
#define CPUAPPLYVISIONROPEFUNC_HPP
#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"

namespace mllm {
class Tensor;

class CPUApplyVisionRoPEFunction : public TensorFunction {
    void rope_hf(Tensor *input, Tensor *rotary_pos_emb, Tensor *output,
                 int thread_count=4) {
        auto out_dtype = output->dtype();
        int partial_dimension = input->dimension();
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
                                auto rope_d = rotary_pos_emb->dataAt<float>(0,0,s,d);
                                float sin_value = std::sin(rope_d); //sin_[s][d];
                                float cos_value = std::cos(rope_d);//cos_[s][d];
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
                                    auto rope_d = rotary_pos_emb->dataAt<float>(0,0,s,d);
                                    float sin_value = std::sin(rope_d); //sin_[s][d];
                                    float cos_value = std::cos(rope_d);//cos_[s][d];
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
                                    auto rope_d = rotary_pos_emb->dataAt<float>(0,0,s,d);
                                    float sin_value = std::sin(rope_d); //sin_[s][d];
                                    float cos_value = std::cos(rope_d);//cos_[s][d];
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
                            auto rope_d = rotary_pos_emb->dataAt<float>(0,0,s,d);
                            float sin_value = std::sin(rope_d); //sin_[s][d];
                            float cos_value = std::cos(rope_d);//cos_[s][d];
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
                            auto rope_d = rotary_pos_emb->dataAt<float>(0,0,s,d);
                            float sin_value = std::sin(rope_d); //sin_[s][d];
                            float cos_value = std::cos(rope_d);//cos_[s][d];
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

public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype());
        outputs[0]->alloc();
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        auto input = inputs[0];
        auto rotary_pos_emb = inputs[1];
        rope_hf(input, rotary_pos_emb, outputs[0], CPUBackend::cpu_threads);
    }
};
} // namespace mllm
#endif // CPUAPPLYVISIONROPEFUNC_HPP