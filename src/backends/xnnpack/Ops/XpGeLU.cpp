#include "backends/xnnpack/Ops/XpGeLU.hpp"
#include "Types.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

ErrorCode XpGeLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode XpGeLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode XpGeLU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    auto status = xnn_define_gelu(xpb->getCurProcessingGraph()->getXnnSubgraph(), inputs[0]->uuid(), outputs[0]->uuid(), 0);

    if (status != xnn_status_success) {
        Log::error("XpGeLU::execute Error");
        exit(-1);
    }

    return MLLM_NO_ERROR;
}

Op *XpGeLUCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpGeLU(bk, name, thread_count);
}
} // namespace mllm::xnnpack
