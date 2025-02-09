
#include "QNNSplitInput.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

namespace mllm {
QNNSplitInput::QNNSplitInput(Backend *bn, string opName, bool isPrompt, int num) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNSplitInput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (int i = 0; i < inputs.size(); i++) {
        outputs[i]->shallowCopyFrom(inputs[i].get(), true);
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSplitInput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (int i = 0; i < inputs.size(); i++) {
        outputs[i]->shallowCopyFrom(inputs[i].get(), true);
    }
    return MLLM_NO_ERROR;
}

} // namespace mllm
