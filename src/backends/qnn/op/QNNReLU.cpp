
#include "QNNReLU.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>

namespace mllm {
QNNReLU::QNNReLU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNReLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNReLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // Only support QUINT8 ReLU

    if (inputs[0]->dtype() == MLLM_TYPE_I8) {
        outputs[0]->setDtype(MLLM_TYPE_I8);
        outputs[0]->quant_param.scale = inputs[0]->quant_param.scale;
        return graphAddNode(name(), "Relu", inputs, outputs, {}, "qti.aisw", true);
    } else {
        return graphAddNode(name(), "LLaMAReLU", inputs, outputs, {}, "LLaMAPackage", true);
    }
}

} // namespace mllm
