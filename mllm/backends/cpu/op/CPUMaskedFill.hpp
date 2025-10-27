//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUMaskedFill_HPP
#define CPUMaskedFill_HPP

#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>
// #include <queue>
// #include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUMaskedFill : public Op {
private:
    int thread_count = 4;
    float value_;

public:
    CPUMaskedFill(Backend *bn, float value, string name, int threadCount) :
        Op(bn, name), value_(value), thread_count(threadCount) {
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // assert(outputs.size() == 2); // topk returns values and indices
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
        outputs[0]->setDtype(inputs[0]->dtype()); // topk_values

        // 遵从原始 reshape 逻辑，在这里 alloc
        // outputs[0]->alloc();

        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        // std::cout << inputs[0]->name() << " " << inputs[1]->name() << std::endl;
        memcpy(outputs[0]->hostPtr<float>(), inputs[0]->hostPtr<float>(), inputs[0]->cntSize());
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < inputs[0]->batch(); n++) {
            for (int h = 0; h < inputs[0]->head(); h++) {
                for (int s = 0; s < inputs[0]->sequence(); s++) {
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        float mask_flag = inputs[1]->dataAt<float>(n, h, s, d);
                        if ((int)mask_flag == 1) {
                            outputs[0]->setDataAt(n, h, s, d, value_); // MaskedFill operation: negation
                        }
                    }
                }
            }
        }
        // NOTE: Add cases for other dimensions if needed.
        return MLLM_NO_ERROR;
    }
};

class CPUMaskedFillCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        float value = op_param["value"];
        return new CPUMaskedFill(bn, value, name, threadCount);
    }
};

} // namespace mllm
#endif // CPUMaskedFill_HPP