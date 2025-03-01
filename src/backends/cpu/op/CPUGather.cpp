#include "CPUGather.hpp"
#include <vector>

namespace mllm {

CPUGather::CPUGather(Backend *bn,  string opName, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUGather::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    assert(inputs.size() == 3);
    assert(outputs.size() == 1);
    if(inputs[1]->batch() == 0) {
        outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
        return Op::reshape(inputs, outputs);
    }
    assert(inputs[0]->batch() == inputs[1]->batch());
    assert(inputs[0]->head() == inputs[1]->head());
    assert(inputs[0]->head() == 1);
    assert(inputs[0]->dimension() == inputs[1]->dimension());
    assert(inputs[2]->dimension() == 1);
    outputs[0]->reshape(inputs[0]->batch(), 1, inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUGather::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if(inputs[1]->batch() == 0) {
        return Op::execute(inputs, outputs);
    }

    assert(inputs[0]->ctype() == BSHD);
    assert(inputs[1]->ctype() == BSHD);
    assert(outputs[0]->ctype() == BSHD);
    auto input_indices = inputs[2];
    int hiddenSize = inputs[0]->dimension();
    for (int batch = 0; batch < inputs[0]->batch(); ++batch) {
        for (int seq = 0; seq < inputs[0]->sequence(); ++seq) {
            if(input_indices->dataAt<float>(batch, 0, seq, 0) >= 0) {
                memcpy(outputs[0]->hostPtr<float>() + outputs[0]->offset(batch, 0, seq, 0),
                       inputs[1]->hostPtr<float>() + (int)inputs[1]->offset(batch, 0, input_indices->dataAt<float>(batch, 0, seq, 0), 0),
                       inputs[1]->dtypeSize() * hiddenSize);
            }
        }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUGather::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    if(inputs[0]->masterTensor() == nullptr) {
        inputs[0]->free();
    }
    outputs[0]->setDtype(activation_dtype());
    outputs[0]->alloc();
    inputs[0]->shallowCopyFrom(outputs[0].get(), false);
    return MLLM_NO_ERROR;
}
} // namespace mllm

