#include "backends/xnnpack/Ops/XpSiLU.hpp"
#include "Types.hpp"
#include "xnnpack.h"

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
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    // o = x * sigmoid(x)
    std::vector<size_t> tmp_shape{
        (size_t)inputs[0]->batch(),
        (size_t)inputs[0]->sequence(),
        (size_t)inputs[0]->head(),
        (size_t)inputs[0]->dimension(),
    };
    auto tmp = defineTemporaryTensor(xpb->getCurProcessingGraph(), tmp_shape, inputs[0]->dtype());
    auto status = xnn_define_unary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        xnn_unary_sigmoid,
        nullptr,
        inputs[0]->uuid(),
        tmp, 0);

    status = xnn_define_binary(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        xnn_binary_multiply,
        nullptr,
        inputs[0]->uuid(),
        tmp,
        outputs[0]->uuid(),
        0);

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