
#include "CPUMergeOutput.hpp"
#include "Types.hpp"
#include <cstring>

namespace mllm {

CPUMergeOutput::CPUMergeOutput(Backend *bn, string opName, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
}

ErrorCode CPUMergeOutput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == outputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        outputs[i]->setDtype(inputs[i]->dtype());
        outputs[i]->reshape(inputs[i]->batch(), inputs[i]->head(), inputs[i]->sequence(), inputs[i]->dimension());
    }

    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMergeOutput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (int i = 0; i < inputs.size(); i++) {
        if (inputs[i]->device() == MLLM_QNN || (inputs[i]->masterTensor() && inputs[i]->masterTensor()->device() == MLLM_QNN)) {
            outputs[i]->deepCopyFrom(inputs[i].get(), true);
        } else {
            if (inputs[i]->allocted() != 0) inputs[i]->free();
            outputs[i]->alloc();
            inputs[i]->deepCopyFrom(outputs[i].get(), true);
        }
    }

    return MLLM_NO_ERROR;
}

ErrorCode CPUMergeOutput::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::execute(inputs, outputs);
}
} // namespace mllm
