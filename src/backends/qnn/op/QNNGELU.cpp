
#include "QNNGELU.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include "Context.hpp"

namespace mllm {
QNNGELU::QNNGELU(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
    scale_.setBackend(Backend::global_backends[MLLM_CPU].get());
}

ErrorCode QNNGELU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNGELU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // Todo: gelu do not supprt signed fix int8
    for (int i = 0; i < inputs.size(); ++i) {
        outputs[i]->setDtype(inputs[i]->dtype());
    }
    return graphAddNode(name(), "Gelu", inputs, outputs, {}, "qti.aisw", true);
}

} // namespace mllm
