#include "backends/xnnpack/Ops/XpSiLU.hpp"
#include "Types.hpp"

namespace mllm::xnnpack {

ErrorCode XpSiLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode XpSiLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode XpSiLU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb, inputs);
    tryDefineAllXpTensors(xpb, outputs);

    auto status = xnn_define_hardswish(xpb->getXnnSubgraph(), inputs[0]->uuid(), outputs[0]->uuid(), 0);

    if (status != xnn_status_success) {
        Log::error("XpSiLU::execute Error");
        exit(-1);
    }

    return MLLM_NO_ERROR;
}

Op *XpSiLUCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpSiLU(bk, name, thread_count);
}
} // namespace mllm::xnnpack