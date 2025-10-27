//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUTRANSPOSEFUNC_HPP
#define CPUTRANSPOSEFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
// #include "Module.hpp"
#include "CPUBackend.hpp"
#include "compute/Transpose2D.hpp"
#include "backends/cpu/third_party/ggml/Quantize.hpp"
#include <cassert>
// #include <ostream>
#include <vector>
#include <memory>
#include <utility>   // For std::pair
#include <algorithm> // For std::equal

namespace mllm {
class Tensor;

class CPUtransposeFunction : public Op {
private:
    int thread_count = 4;
    vector<std::pair<Chl, Chl>> axiss_;

public:
    CPUtransposeFunction(Backend *bn, string name, int threadCount, const vector<std::pair<Chl, Chl>> &axiss) :
        Op(bn, name), thread_count(threadCount), axiss_(axiss) {
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // for BSHD attention start
        if (axiss_.size() == 1 && axiss_[0].first == HEAD && axiss_[0].second == SEQUENCE) {
            if (inputs[0]->ctype() == BSHD) {
                outputs[0]->chls() = {{BATCH, 0}, {HEAD, 1}, {SEQUENCE, 2}, {DIMENSION, 3}};
            } else { // inputs[0]->ctype() == BHSD
                outputs[0]->chls() = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}};
            }
            outputs[0]->changeCtype(4);
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
            return MLLM_NO_ERROR;
        } else if (axiss_.size() == 1 && axiss_[0].first == SEQUENCE && axiss_[0].second == DIMENSION && inputs[0]->ctype() == BHSD) {
            outputs[0]->setCtype(BHSD);
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[0]->sequence());
            return MLLM_NO_ERROR;
        }
        // for BSHD attention end

        if (!outputs[0]->undiffusion()) {
            outputs[0]->transCopyShape(inputs[0]->shape());
            std::map<Chl, int> origin_chls = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
            if (std::equal(outputs[0]->chls().begin(), outputs[0]->chls().end(), origin_chls.begin())) {
                outputs[0]->chls() = inputs[0]->chls();
                for (auto axis : axiss_) {
                    auto axis0 = axis.first;
                    auto axis1 = axis.second;
                    auto ori_0_idx = outputs[0]->chls()[axis0];
                    auto ori_1_idx = outputs[0]->chls()[axis1];
                    outputs[0]->chls()[axis0] = ori_1_idx;
                    outputs[0]->chls()[axis1] = ori_0_idx;
                }
                outputs[0]->changeCtype(inputs[0]->shape().size());
                outputs[0]->undiffusion() = true;
            }
        }

        if (inputs[0]->masterTensor() != nullptr && (inputs[0]->masterTensor()->name().find("Cache") != std::string::npos || inputs[0]->masterTensor()->name().find("weight") != std::string::npos)) {
            if (outputs[0]->masterTensor() == nullptr) {
                outputs[0]->setDtype(inputs[0]->dtype());
                outputs[0]->shallowCopyFrom(inputs[0], false);
            }
        } else {
            if (inputs[0]->masterTensor() == nullptr) {
                inputs[0]->free();
            }
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->alloc();
            inputs[0]->setUndiffusion(true);
            inputs[0]->shallowCopyFrom(outputs[0], false);
            outputs[0]->transFrom() = axiss_;
        }
        return MLLM_NO_ERROR;
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // for BSHD attention start
        if (axiss_.size() == 1 && axiss_[0].first == HEAD && axiss_[0].second == SEQUENCE) {
            outputs[0]->transCopyShape(inputs[0]->shape());
            outputs[0]->chls() = inputs[0]->chls();
            std::swap(outputs[0]->chls()[HEAD], outputs[0]->chls()[SEQUENCE]);
            outputs[0]->changeCtype(inputs[0]->shape().size());
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
            return MLLM_NO_ERROR;
        } else if (axiss_.size() == 1 && axiss_[0].first == SEQUENCE && axiss_[0].second == DIMENSION && inputs[0]->ctype() == BHSD) {
            outputs[0]->setCtype(BHSD);
            outputs[0]->setDtype(inputs[0]->dtype());
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[0]->sequence());
            return MLLM_NO_ERROR;
        }
        // for BSHD attention end

        std::map<Chl, int> origin_chls = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
        auto origin_s = inputs[0]->shape().size();
        outputs[0]->transCopyShape(inputs[0]->shape());

        if (inputs[0]->masterTensor() == nullptr || std::equal(outputs[0]->chls().begin(), outputs[0]->chls().end(), origin_chls.begin())) {
            outputs[0]->chls() = inputs[0]->chls();
            for (auto axis : axiss_) {
                auto axis0 = axis.first;
                auto axis1 = axis.second;
                std::swap(outputs[0]->chls()[axis0], outputs[0]->chls()[axis1]);
            }
            outputs[0]->changeCtype(origin_s);
            outputs[0]->undiffusion() = true;
        }

        if (inputs[0]->masterTensor() != nullptr && (inputs[0]->masterTensor()->name().find("Cache") != std::string::npos || inputs[0]->masterTensor()->name().find("weight") != std::string::npos)) {
            if (outputs[0]->masterTensor() == nullptr) {
                outputs[0]->setDtype(inputs[0]->dtype());
                outputs[0]->shallowCopyFrom(inputs[0], false);
            }
        }
        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // for BSHD attention start
        if (axiss_.size() == 1 && axiss_[0].first == HEAD && axiss_[0].second == SEQUENCE) {
            // This is a physical transpose, allocate and copy
            if (outputs[0]->hostPtr<void>() == nullptr) {
                outputs[0]->alloc();
            }
            // BSHD -> BHSD (transpose S and H)
            // 真转置
            assert(inputs[0]->batch() == 1);
            assert(outputs[0]->batch() == 1);
            assert(inputs[0]->head() == outputs[0]->head());
            assert(inputs[0]->sequence() == outputs[0]->sequence());
            if (inputs[0]->dtype() == outputs[0]->dtype()) {
#pragma omp parallel for num_threads(thread_count)
                for (int h = 0; h < inputs[0]->head(); ++h) {
                    for (int s = 0; s < inputs[0]->sequence(); ++s) {
                        auto input_ptr = inputs[0]->ptrAt<float>(0, h, s, 0);
                        auto output_ptr = outputs[0]->ptrAt<float>(0, h, s, 0);
                        memcpy(output_ptr, input_ptr, inputs[0]->dimension() * sizeof(float));
                    }
                }
            } else { // With quantization
#pragma omp parallel for num_threads(thread_count)
                for (int h = 0; h < inputs[0]->head(); ++h) {
                    for (int s = 0; s < inputs[0]->sequence(); ++s) {
                        // auto input_ptr = inputs[0]->ptrAt<float>(0, h, s, 0);
                        // auto output_ptr = outputs[0]->ptrAt<mllm_fp16_t>(0, h, s, 0);
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            // output_ptr[d] = MLLM_FP32_TO_FP16(input_ptr[d]);
                            auto value = inputs[0]->dataAt<float>(0, h, s, d);
                            outputs[0]->setDataAt<mllm_fp16_t>(0, h, s, d, MLLM_FP32_TO_FP16(value));
                        }
                    }
                }
            }

        } else if (axiss_.size() == 1 && axiss_[0].first == SEQUENCE && axiss_[0].second == DIMENSION && inputs[0]->ctype() == BHSD) {
            assert(outputs[0]->ctype() == BHSD);
            // 真转置
            assert(inputs[0]->batch() == 1);
            assert(outputs[0]->batch() == 1);
            assert(inputs[0]->sequence() == outputs[0]->dimension());
            assert(outputs[0]->sequence() == inputs[0]->dimension());
            // BHSD->BHDS
            const int N = inputs[0]->sequence();
            const int M = inputs[0]->dimension();
            if (inputs[0]->dtype() == MLLM_TYPE_F32) {
#pragma omp parallel for num_threads(thread_count)
                for (int h = 0; h < inputs[0]->head(); ++h) {
                    const float *src_ptr = inputs[0]->ptrAt<float>(0, h, 0, 0);
                    float *dst_ptr = outputs[0]->ptrAt<float>(0, h, 0, 0);
                    transpose_matrix_efficient(src_ptr, dst_ptr, N, M);
                }
            } else {
#if defined(__aarch64__)
#pragma omp parallel for num_threads(thread_count)
                for (int h = 0; h < inputs[0]->head(); ++h) {
                    const mllm_fp16_t *src_ptr = inputs[0]->ptrAt<mllm_fp16_t>(0, h, 0, 0);
                    mllm_fp16_t *dst_ptr = outputs[0]->ptrAt<mllm_fp16_t>(0, h, 0, 0);
                    transpose_matrix_efficient_fp16(src_ptr, dst_ptr, N, M);
                }
#else
                std::cout << "FP16 transpose not supported on non-aarch64 platform" << std::endl;
#endif
            }
        }
        // for BSHD attention end
        // Note: The general transpose case is handled by metadata changes in reshape/setUp
        // and does not require data movement in execute.
        return MLLM_NO_ERROR;
    }
};

class CPUtransposeFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        // Assumes OpParam is structured to pass the axis pairs.
        // Example: {"num_pairs": 1, "axis1_0": 2, "axis2_0": 1} (HEAD, SEQUENCE)
        int num_pairs = static_cast<int>(op_param.at("num_pairs"));
        vector<std::pair<Chl, Chl>> axiss;
        for (int i = 0; i < num_pairs; ++i) {
            Chl axis1 = (Chl)op_param.at("axis1_" + std::to_string(i));
            Chl axis2 = (Chl)op_param.at("axis2_" + std::to_string(i));
            axiss.push_back({axis1, axis2});
        }
        return new CPUtransposeFunction(bn, name, threadCount, axiss);
    }
};

} // namespace mllm
#endif // CPUTRANSPOSEFUNC_HPP