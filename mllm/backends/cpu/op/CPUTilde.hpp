//
// Created by Rongjie Yi on 24-12-16.
//

#ifndef CPUTILDE_HPP
#define CPUTILDE_HPP

#include "CPUBackend.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include <cassert>
// #include <queue>
// #include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUTilde : public Op {
private:
    int thread_count = 4;

public:
    CPUTilde(Backend *bn, string name, int threadCount) :
        Op(bn, name), thread_count(threadCount) {
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
#pragma omp parallel for collapse(4) num_threads(CPUBackend::cpu_threads)
        for (int n = 0; n < inputs[0]->batch(); n++) {
            for (int h = 0; h < inputs[0]->head(); h++) {
                for (int s = 0; s < inputs[0]->sequence(); s++) {
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        float value = inputs[0]->dataAt<float>(n, h, s, d);
                        assert(((int)value == 1 || (int)value == 0) && "Tilde operation expects input to be 1.0");
                        float set_data = ((int)value == 1) ? 0.0F : 1.0F; // Tilde operation: negation
                        outputs[0]->setDataAt(n, h, s, d, set_data);      // Tilde operation: negation
                    }
                }
            }
        }
        return MLLM_NO_ERROR;
    }
};

class CPUTildeCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUTilde(bn, name, threadCount);
    }
};

} // namespace mllm
#endif // CPUTILDE_HPP