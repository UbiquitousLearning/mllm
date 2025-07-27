//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUTOPKFUNC_HPP
#define CPUTOPKFUNC_HPP

#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>
#include <queue>
#include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUtopkFunction : public Op {
private:
    int thread_count = 4;
    int k_;
    Chl dim_;

public:
    CPUtopkFunction(Backend *bn, string name, int threadCount, int k, Chl dim) :
        Op(bn, name), thread_count(threadCount), k_(k), dim_(dim) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        assert(outputs.size() == 2); // topk returns values and indices
        if (dim_ == DIMENSION) {
            outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), k_);
            outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), k_);
        } else if (dim_ == HEAD) {
            assert(inputs[0]->dimension() == 1 && "Only support topk on last dimension currently.");
            outputs[0]->reshape(inputs[0]->batch(), k_, inputs[0]->sequence(), 1); // topk values
            outputs[1]->reshape(inputs[0]->batch(), k_, inputs[0]->sequence(), 1); // topk values
        }
        // NOTE: Add cases for other dimensions if needed.

        outputs[0]->setDtype(inputs[0]->dtype()); // topk_values
        outputs[1]->setDtype(inputs[0]->dtype()); // topk_indices are typically int, but float is used here

        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();
        // outputs[1]->alloc();

        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (dim_ == DIMENSION) {
#pragma omp parallel for collapse(3) num_threads(CPUBackend::cpu_threads)
            for (int n = 0; n < inputs[0]->batch(); n++) {
                for (int h = 0; h < inputs[0]->head(); h++) {
                    for (int s = 0; s < inputs[0]->sequence(); s++) {
                        std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> topk_value_indices;
                        for (int d = 0; d < inputs[0]->dimension(); ++d) {
                            float value = inputs[0]->dataAt<float>(n, h, s, d);
                            topk_value_indices.push({value, d});
                            if (topk_value_indices.size() > k_) {
                                topk_value_indices.pop();
                            }
                        }
                        for (int d = k_ - 1; d >= 0; --d) {
                            auto top = topk_value_indices.top();
                            topk_value_indices.pop();
                            outputs[0]->setDataAt<float>(n, h, s, d, top.first);
                            outputs[1]->setDataAt<float>(n, h, s, d, top.second);
                        }
                    }
                }
            }
        } else if (dim_ == HEAD) {
            for (int n = 0; n < inputs[0]->batch(); n++) {
                for (int s = 0; s < inputs[0]->sequence(); s++) {
                    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<>> topk_value_indices;
                    for (int h = 0; h < inputs[0]->head(); h++) {
                        float value = inputs[0]->dataAt<float>(n, h, s, 0);
                        topk_value_indices.push({value, h});
                        if (topk_value_indices.size() > k_) {
                            topk_value_indices.pop();
                        }
                    }
                    for (int h = k_ - 1; h >= 0; --h) {
                        auto top = topk_value_indices.top();
                        topk_value_indices.pop();
                        outputs[0]->setDataAt<float>(n, h, s, 0, top.first);
                        outputs[1]->setDataAt<float>(n, h, s, 0, top.second);
                    }
                }
            }
        }
        // NOTE: Add cases for other dimensions if needed.
        return MLLM_NO_ERROR;
    }
};

class CPUtopkFunctionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        int k = static_cast<int>(op_param.at("k"));
        Chl dim = (Chl)op_param.at("dim");
        return new CPUtopkFunction(bn, name, threadCount, k, dim);
    }
};

} // namespace mllm
#endif // CPUTOPKFUNC_HPP