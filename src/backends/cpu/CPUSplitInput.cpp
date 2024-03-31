
#include "CPUSplitInput.hpp"
#include <cstring>

namespace mllm {

CPUSplitInput::CPUSplitInput(Backend *bn, string opName, bool isPrompt, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    isPrompt_ = isPrompt;
}

ErrorCode CPUSplitInput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 3);

    if (isPrompt_) {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 3, inputs[0]->dimension());
        outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 3, inputs[0]->dimension());
        outputs[2]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 3, inputs[0]->dimension());

    } else {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), 1, inputs[0]->dimension());
        outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), (inputs[0]->sequence() - 1) / 2, inputs[0]->dimension());
        outputs[2]->reshape(inputs[0]->batch(), inputs[0]->head(), (inputs[0]->sequence() - 1) / 2, inputs[0]->dimension());
    }

    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSplitInput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    activation_dtype_ = inputs[0]->dtype();
    return Op::setUp(inputs, outputs);
}

ErrorCode CPUSplitInput::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "CPUSplitInput::execute" << std::endl;
    std::cout << inputs[0]->dtype() << std::endl;
    std::cout << outputs[0]->dtype() << std::endl;
    std::cout << outputs[1]->dtype() << std::endl;
    std::cout << outputs[2]->dtype() << std::endl;
    

    // copy data from input to output
    int offset = 0;
    memcpy(outputs[0]->hostPtr<void>(), inputs[0]->hostPtr<void>(), outputs[0]->cntSize());
    offset += outputs[0]->cntSize();
    memcpy(outputs[1]->hostPtr<void>(), (bool*)inputs[0]->hostPtr<void>() + offset, outputs[1]->cntSize());
    offset += outputs[1]->cntSize();
    memcpy(outputs[2]->hostPtr<void>(), (bool*)inputs[0]->hostPtr<void>() + offset, outputs[2]->cntSize());
    return Op::execute(inputs, outputs);
}
} // namespace mllm

