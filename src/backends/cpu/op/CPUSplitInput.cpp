
#include "CPUSplitInput.hpp"
#include <cstring>

namespace mllm {

CPUSplitInput::CPUSplitInput(Backend *bn, string opName, bool isPrompt, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    isPrompt_ = isPrompt;
}

ErrorCode CPUSplitInput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (int i = 0; i < inputs.size(); i++) {
        outputs[i]->reshape(inputs[i]->batch(), inputs[i]->head(), inputs[i]->sequence(), inputs[i]->dimension());
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSplitInput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (int i = 0; i < inputs.size(); i++) {
        outputs[i]->shallowCopyFrom(inputs[i].get(), true);
        // the split output is CPU backend by default, set output backend to QNN to let the device() be QNN
        outputs[i]->setBackend(inputs[i]->backend());

    }
    return MLLM_NO_ERROR;
}

ErrorCode CPUSplitInput::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::execute(inputs, outputs);
}
} // namespace mllm

