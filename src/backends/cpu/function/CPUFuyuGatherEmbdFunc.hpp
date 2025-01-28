//
// Created by Rongjie Yi on 24-2-26.
//

#ifndef CPUFUYUGATHEREMBDFUNC_HPP
#define CPUFUYUGATHEREMBDFUNC_HPP
#include "Tensor.hpp"
#include "Types.hpp"

namespace mllm {
class Tensor;

class CPUFuyuGatherEmbdFunc : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        assert(inputs.size() == 3);
        assert(outputs.size() == 1);
        if (inputs[1]->batch() == 0) {
            outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
        }
        assert(inputs[0]->batch() == inputs[1]->batch());
        assert(inputs[0]->head() == inputs[1]->head());
        assert(inputs[0]->head() == 1);
        assert(inputs[0]->dimension() == inputs[1]->dimension());
        assert(inputs[2]->dimension() == 1);
        outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
        // alloc

        if (inputs[0]->masterTensor() == nullptr) {
            inputs[0]->free();
        }
        outputs[0]->alloc();
        inputs[0]->shallowCopyFrom(outputs[0], false);
    }
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override {
        if (inputs[1]->batch() == 0) {
            return;
        }
        assert(inputs[0]->ctype() == BSHD);
        assert(inputs[1]->ctype() == BSHD);
        assert(outputs[0]->ctype() == BSHD);
        auto input_indices = inputs[2];
        int hiddenSize = inputs[0]->dimension();
        for (int batch = 0; batch < inputs[0]->batch(); ++batch) {
            for (int seq = 0; seq < inputs[0]->sequence(); ++seq) {
                if (input_indices->dataAt<float>(batch, 0, seq, 0) >= 0) {
                    memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(batch, 0, seq, 0),
                           inputs[1]->hostPtr<float>() + (int)inputs[1]->offset(batch, 0, input_indices->dataAt<float>(batch, 0, seq, 0), 0),
                           inputs[1]->dtypeSize() * hiddenSize);
                }
            }
        }
    }
};

} // namespace mllm
#endif // CPUFUYUGATHEREMBDFUNC_HPP