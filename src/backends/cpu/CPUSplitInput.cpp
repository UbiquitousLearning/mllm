
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
    assert(outputs.size() == 4);

    if (isPrompt_) {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 7, inputs[0]->dimension());
        outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 7, inputs[0]->dimension());
        outputs[2]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 7, inputs[0]->dimension());

        // do not * 4 since type is FP32
        outputs[3]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 7, inputs[0]->dimension());

    } else {
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), 1, inputs[0]->dimension());
        outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), (inputs[0]->sequence() - 1) / 3, inputs[0]->dimension());
        outputs[2]->reshape(inputs[0]->batch(), inputs[0]->head(), (inputs[0]->sequence() - 1) / 3, inputs[0]->dimension());
        outputs[3]->reshape(inputs[0]->batch(), inputs[0]->head(), (inputs[0]->sequence() - 1) / 3, inputs[0]->dimension());
    }

    return Op::reshape(inputs, outputs);
}

ErrorCode CPUSplitInput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    activation_dtype_ = inputs[0]->dtype();
    // return Op::setUp(inputs, outputs);

    for ( int i = 0; i<outputs.size(); i++) {
        if (i < 3)
            outputs[i]->setDtype(activation_dtype_);
        else
            outputs[i]->setDtype(MLLM_TYPE_F32);
        outputs[i]->alloc();
    }
    return MLLM_NO_ERROR; 
}

ErrorCode CPUSplitInput::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // copy data from input to output
    int offset = 0;
    memcpy(outputs[0]->hostPtr<void>(), inputs[0]->hostPtr<void>(), outputs[0]->cntSize());
    offset += outputs[0]->cntSize();
    memcpy(outputs[1]->hostPtr<void>(), inputs[0]->hostPtr<uint8_t>() + offset, outputs[1]->cntSize());
    offset += outputs[1]->cntSize();
    memcpy(outputs[2]->hostPtr<void>(), inputs[0]->hostPtr<uint8_t>() + offset, outputs[2]->cntSize());
    offset += outputs[2]->cntSize();
    memcpy(outputs[3]->hostPtr<void>(), inputs[0]->hostPtr<uint8_t>() + offset, outputs[3]->cntSize());

    return Op::execute(inputs, outputs);
}
} // namespace mllm

