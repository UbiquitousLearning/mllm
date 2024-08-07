
#include "QNNMergeOutput.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

#define DYNAMICBUFFER 32

namespace mllm {
QNNMergeOutput::QNNMergeOutput(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNMergeOutput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 4 || inputs.size() == 3);
    assert(outputs.size() == 1);

    if (inputs.size() == 3)
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() + inputs[1]->sequence() + inputs[2]->sequence(), inputs[0]->dimension());
    else
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), (inputs[0]->sequence() * 3 + inputs[3]->sequence()) * 4, inputs[0]->dimension());

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMergeOutput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode QNNMergeOutput::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->free();

    return MLLM_NO_ERROR;
}

ErrorCode QNNMergeOutput::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs.size() == 4) {
        memcpy(outputs[0]->hostPtr<uint8_t>() + (inputs[0]->cntSize() * 3), inputs[3]->hostPtr<uint8_t>(), inputs[3]->cntSize());
    }

    return MLLM_NO_ERROR;
}

} // namespace mllm
