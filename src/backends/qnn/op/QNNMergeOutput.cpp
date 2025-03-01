
#include "QNNMergeOutput.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <memory>

#define DYNAMICBUFFER 32

namespace mllm {
QNNMergeOutput::QNNMergeOutput(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNMergeOutput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // deepCopy at reshape to let QNNCommonOp::setUp to get the correct ttype
    for(int i = 0; i < inputs.size(); i++) {
        outputs[i]->shallowCopyFrom(inputs[i].get(), true);
    }

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNMergeOutput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (int i = 0; i < inputs.size(); i++) {
        outputs[i]->shallowCopyFrom(inputs[i].get(), true);
    }
    return MLLM_NO_ERROR;
}

ErrorCode QNNMergeOutput::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->free();

    return MLLM_NO_ERROR;
}

} // namespace mllm
