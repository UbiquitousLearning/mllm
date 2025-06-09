
#include "CPURoPE.hpp"
#include "CPURoPETree.hpp"
#include "Timing.hpp"
#include "Types.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include "backends/cpu/quantize/QuantizeQ8.hpp"

namespace mllm {

vector<float> CPURoPETree::theta_;

vector<vector<float>> CPURoPETree::sin_;
vector<vector<float>> CPURoPETree::cos_;
int CPURoPETree::global_pose_type_ = -1;
int CPURoPETree::ishape_old;

CPURoPETree::CPURoPETree(Backend *bn, string opName, int pose_type, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
}

CPURoPETree::CPURoPETree(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
    rope_theta_ = rope_theta;
    pos_max_ = max_position_embeddings;
}

CPURoPETree::CPURoPETree(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    pose_type_ = pose_type;
    rope_theta_ = rope_theta;
    partial_rotary_factor_ = partial_rotary_factor;
    pos_max_ = max_position_embeddings;
}

CPURoPETree::CPURoPETree(Backend *bn, string opName, OpParam &config, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    config_ = config;
    pose_type_ = config.at("pose_type");
    auto it = config.find("rope_theta");
    if (it != config.end()) {
        rope_theta_ = it->second;
    }
    it = config.find("partial_rotary_factor");
    if (it != config.end()) {
        partial_rotary_factor_ = it->second;
    }
    it = config.find("max_position_embeddings");
    if (it != config.end()) {
        pos_max_ = it->second;
    }
    rope_type = (RoPEThetaType)config.at("rope_type");
}

ErrorCode CPURoPETree::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 2);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    ishape = inputs[0]->dimension() * partial_rotary_factor_;
    // pos_max_ = 16384;

    if (sin_.empty() || ishape_old < ishape || global_pose_type_ != pose_type_) {
        auto calc_theta = rope_init_func_map.at(rope_type);
        auto config = config_;
        config["base"] = (float)rope_theta_;
        config["dim"] = ishape;
        float attention_scaling = calc_theta(config, theta_);

        global_pose_type_ = pose_type_;
        ishape_old = ishape;
        if (pose_type_ == LLAMAROPE) {
            sinusoidal_position_embedding_llama(pos_max_, ishape, theta_, sin_, cos_, attention_scaling);
        } else if (pose_type_ == PERSIMMONROPE) {
            sinusoidal_position_embedding_huggingface(pos_max_, ishape / 2, theta_, sin_, cos_, attention_scaling);
        } else if (pose_type_ == HFHUBROPE || pose_type_ == MLAROPE) {
            sinusoidal_position_embedding_huggingface(pos_max_, ishape, theta_, sin_, cos_, attention_scaling);
        } else {
        }
    }
#ifdef USE_QNN
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
    if (cpuBackend->isStageSwitching()) {
        h_cnt_ = cpuBackend->getCurSequenceLength();
    }
#else
    auto cpuBackend = dynamic_cast<CPUBackend *>(backend_);
#endif

    // for sd
    if (cpuBackend->isUsingDraft()) {
        unsigned int last_draft_length = cpuBackend->getLastDraftLength();
        const std::vector<unsigned int> &last_verified_position_ids = cpuBackend->getLastVerifiedPositionIds();
        h_cnt_ = h_cnt_ - (last_draft_length) + last_verified_position_ids.size();
        if (h_cnt_ < 0) {
            h_cnt_ = 0;
        }
    }
    return Op::reshape(inputs, outputs);
}

void CPURoPETree::rope_llama(shared_ptr<Tensor> input, shared_ptr<Tensor> output) {
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequance
                for (int d = 0; d < partial_dimension; d += 2) {
                    float in_value = input->dataAt<float>(n, h, s, d);
                    float in_value_2 = input->dataAt<float>(n, h, s, d + 1);
                    float sin_value = sin_[s + h_cnt_][d];
                    float cos_value = cos_[s + h_cnt_][d];
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

void CPURoPETree::rope_hf(shared_ptr<Tensor> input, shared_ptr<Tensor> output, shared_ptr<Tensor> tree_ancestor) {
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
    int half = (int)(partial_dimension / 2);
    assert(partial_dimension % 2 == 0);
    std::vector<unsigned int> position_ids(tree_ancestor->sequence());
    if (output->ctype() == BSHD) {
        if (input->dtype() == MLLM_TYPE_F16) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
            for (int n = 0; n < input->batch(); ++n) {
                for (int h = 0; h < input->head(); ++h) {
                    for (int s = 0; s < input->sequence(); ++s) { // sequance
                        int pos = 0;
                        if (s == 0 || tree_ancestor->sequence() == 1) {
                            pos = s + h_cnt_;
                            if (tree_ancestor->sequence() > 1) {
                                position_ids[s] = pos;
                            }
                        } else {
                            auto ancestor_idx = tree_ancestor->dataAt<int32_t>(0, 0, s, 0);
                            pos = position_ids[ancestor_idx] + 1;
                            position_ids[s] = pos;
                        }
                        for (int d = 0; d < partial_dimension / 2; ++d) {
                            auto v = input->ptrAt<mllm_fp16_t>(n, h, s, d);
                            auto o = output->ptrAt<mllm_fp16_t>(n, h, s, d);
                            float in_value = static_cast<float>(v[0]);
                            float in_value_2 = static_cast<float>(v[half]);
                            float sin_value = sin_[pos][d];
                            float cos_value = cos_[pos][d];
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
#pragma omp parallel for collapse(2) num_threads(thread_count)
                for (int n = 0; n < input->batch(); ++n) {
                    for (int h = 0; h < input->head(); ++h) {
                        for (int s = 0; s < input->sequence(); ++s) { // sequance
                            int pos = 0;
                            if (s == 0 || tree_ancestor->sequence() == 1) {
                                pos = s + h_cnt_;
                                if (tree_ancestor->sequence() > 1) {
                                    position_ids[s] = pos;
                                }
                            } else {
                                auto ancestor_idx = tree_ancestor->dataAt<int32_t>(0, 0, s, 0);
                                pos = position_ids[ancestor_idx] + 1;
                                position_ids[s] = pos;
                            }
                            for (int d = 0; d < partial_dimension / 2; ++d) {
                                auto v = input->ptrAt<float>(n, h, s, d);
                                auto o = output->ptrAt<float>(n, h, s, d);
                                float in_value = v[0];
                                float in_value_2 = v[half];
                                float sin_value = sin_[pos][d];
                                float cos_value = cos_[pos][d];
                                auto value = in_value * cos_value - in_value_2 * sin_value;
                                auto value2 = in_value * sin_value + in_value_2 * cos_value;
                                o[0] = value;
                                o[half] = value2;
                            }
                        }
                    }
                }
            } else if (out_dtype == MLLM_TYPE_F16) {
#pragma omp parallel for collapse(2) num_threads(thread_count)
                for (int n = 0; n < input->batch(); ++n) {
                    for (int h = 0; h < input->head(); ++h) {
                        for (int s = 0; s < input->sequence(); ++s) { // sequance
                            int pos = 0;
                            if (s == 0 || tree_ancestor->sequence() == 1) {
                                pos = s + h_cnt_;
                                if (tree_ancestor->sequence() > 1) {
                                    position_ids[s] = pos;
                                }
                            } else {
                                auto ancestor_idx = tree_ancestor->dataAt<int32_t>(0, 0, s, 0);
                                pos = position_ids[ancestor_idx] + 1;
                                position_ids[s] = pos;
                            }
                            for (int d = 0; d < partial_dimension / 2; ++d) {
                                auto v = input->ptrAt<float>(n, h, s, d);
                                auto o = output->ptrAt<mllm_fp16_t>(n, h, s, d);
                                float in_value = v[0];
                                float in_value_2 = v[half];
                                float sin_value = sin_[pos][d];
                                float cos_value = cos_[pos][d];
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
#pragma omp parallel for collapse(3) num_threads(thread_count)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequance
                int pos = 0;
                if (s == 0 || tree_ancestor->sequence() == 1) {
                    pos = s + h_cnt_;
                    if (tree_ancestor->sequence() > 1) {
                        position_ids[s] = pos;
                    }
                } else {
                    auto ancestor_idx = tree_ancestor->dataAt<int32_t>(0, 0, s, 0);
                    pos = position_ids[ancestor_idx] + 1;
                    position_ids[s] = pos;
                }
                for (int d = 0; d < partial_dimension / 2; ++d) {
                    if (input->dtype() == MLLM_TYPE_F16) {
                        float in_value = static_cast<float>(input->dataAt<mllm_fp16_t>(n, h, s, d));
                        float in_value_2 = static_cast<float>(input->dataAt<mllm_fp16_t>(n, h, s, d + partial_dimension / 2));
                        float sin_value = sin_[pos][d];
                        float cos_value = cos_[pos][d];
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
                        float sin_value = sin_[pos][d];
                        float cos_value = cos_[pos][d];
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
void CPURoPETree::rope_permission(shared_ptr<Tensor> input, shared_ptr<Tensor> output) {
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
#pragma omp parallel for collapse(4) num_threads(thread_count)
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) { // sequance
                for (int d = 0; d < partial_dimension; ++d) {
                    float in_value = input->dataAt<float>(n, h, s, d);
                    float in_value_2;
                    float sin_value = sin_[s + h_cnt_][d];
                    float cos_value = cos_[s + h_cnt_][d];
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
void CPURoPETree::rope_mla(shared_ptr<Tensor> input, shared_ptr<Tensor> output) {
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
                    float sin_value = sin_[s + h_cnt_][d];
                    float cos_value = cos_[s + h_cnt_][d];
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
// TODO: 理论上Q8这里会调用doExecute(inputs, {tmp_out})，所以我直接改了doExecute应该没有问题
ErrorCode CPURoPETree::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
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
ErrorCode CPURoPETree::doExecute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto &input = inputs[0];
    auto &output = outputs[0];
    auto &tree_ancestor = inputs[1];
    auto out_dtype = output->dtype();
    int partial_dimension = (input->dimension()) * partial_rotary_factor_;
    // auto start_t = mllm_time_us();
    if (pose_type_ == LLAMAROPE) {
        throw std::runtime_error("LLAMAROPE is not implemented for RoPETree");
        rope_llama(input, output);
    } else if (pose_type_ == HFHUBROPE) {
        rope_hf(input, output, tree_ancestor);
    } else if (pose_type_ == PERSIMMONROPE) {
        throw std::runtime_error("PERSIMMONROPE is not implemented for RoPETree");
        rope_permission(input, output);
    } else if (pose_type_ == MLAROPE) {
        throw std::runtime_error("MLAROPE is not implemented for RoPETree");
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
// ErrorCode CPURoPETree::updateVerifiedRoPECache(unsigned int withdrawn_length) {
//     h_cnt_ -= withdrawn_length;
//     if (h_cnt_ < 0) {
//         h_cnt_ = 0;
//     }
//     return MLLM_NO_ERROR;
// }

ErrorCode CPURoPETree::load(AbstructLoader &loader) {
    return Op::load(loader);
}
ErrorCode CPURoPETree::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}
} // namespace mllm
