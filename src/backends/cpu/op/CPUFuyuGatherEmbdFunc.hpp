//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUFUYUGATHEREMBDFUNC_HPP
#define CPUFUYUGATHEREMBDFUNC_HPP

#include "Tensor.hpp"
#include "Types.hpp"
#include "CPUBackend.hpp"
#include <vector>
#include <memory>

namespace mllm {
class Tensor;

class CPUFuyuGatherEmbdFunc : public Op {
private:
    int thread_count = 4;

public:
    CPUFuyuGatherEmbdFunc(Backend *bn, string name, int threadCount) :
        Op(bn, name), thread_count(threadCount) {
    }

    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[0]->masterTensor() == nullptr) {
            inputs[0]->free();
        }
        outputs[0]->alloc();
        inputs[0]->shallowCopyFrom(outputs[0], false);
        return MLLM_NO_ERROR;
    }

    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        assert(inputs.size() == 3);
        assert(inputs[0]->batch() == inputs[1]->batch());
        assert(inputs[0]->head() == inputs[1]->head());
        assert(inputs[0]->head() == 1);
        assert(inputs[0]->dimension() == inputs[1]->dimension());
        assert(inputs[2]->dimension() == 1);

        return MLLM_NO_ERROR;
    }

    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override {
        if (inputs[1]->batch() == 0) {
            return MLLM_NO_ERROR;
        }
        assert(inputs[0]->ctype() == BSHD);
        assert(inputs[1]->ctype() == BSHD);

        auto input_indices = inputs[2];
        int hiddenSize = inputs[0]->dimension();
        for (int batch = 0; batch < inputs[0]->batch(); ++batch) {
            for (int seq = 0; seq < inputs[0]->sequence(); ++seq) {
                if (input_indices->dataAt<float>(batch, 0, seq, 0) >= 0) {
                    memcpy(inputs[0]->hostPtr<float>() + inputs[0]->offset(batch, 0, seq, 0),
                           inputs[1]->hostPtr<float>() + (int)inputs[1]->offset(batch, 0, input_indices->dataAt<float>(batch, 0, seq, 0), 0),
                           inputs[1]->dtypeSize() * hiddenSize);
                }
            }
        }
        return MLLM_NO_ERROR;
    }
};

class CPUFuyuGatherEmbdFuncCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUFuyuGatherEmbdFunc(bn, name, threadCount);
    }
};

} // namespace mllm
#endif // CPUFUYUGATHEREMBDFUNC_HPP