#include "backends/xnnpack/Ops/XpReLU.hpp"
#include "Types.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

ErrorCode XpReLU::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do nothing
    return MLLM_NO_ERROR;
}

ErrorCode XpReLU::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode XpReLU::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    // when negative_slope = 0, leaky_relu is same as relu
    auto status = xnn_define_leaky_relu(xpb->getCurProcessingGraph()->getXnnSubgraph(), 0.f, inputs[0]->uuid(), outputs[0]->uuid(), 0);

    if (status != xnn_status_success) {
        Log::error("XpReLU::execute Error");
        exit(-1);
    }

    return MLLM_NO_ERROR;
}

Op *XpReLUCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpReLU(bk, name, thread_count);
}
} // namespace mllm::xnnpack