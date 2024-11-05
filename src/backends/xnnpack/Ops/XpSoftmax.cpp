#include "backends/xnnpack/Ops/XpSoftmax.hpp"
#include "Types.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

ErrorCode XpSoftmax::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (axis_ != DIMENSION) {
        Log::error("XpSoftmax only support axis=DIMENTSION");
        exit(-1);
    }

    if (do_causal_mask_) {
        Log::error("XpSoftmax will not fuse causal mask operation for you. Pls using explicit causal mask instead");
    }

    return MLLM_NO_ERROR;
}

ErrorCode XpSoftmax::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    return Op::reshape(inputs, outputs);
}

ErrorCode XpSoftmax::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)inputs[0]->backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    // always perform softmax on the last dim.
    auto status = xnn_define_softmax(xpb->getCurProcessingGraph()->getXnnSubgraph(), inputs[0]->uuid(), outputs[0]->uuid(), 0);

    if (status != xnn_status_success) {
        Log::error("XpSoftmax::execute Error");
        exit(-1);
    }

    return MLLM_NO_ERROR;
}

Op *XpSoftmaxCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    int axis = static_cast<int>(op_param["axis"]);
    bool do_causal_mask = static_cast<bool>(op_param["do_causal_mask"]);
    return new XpSoftmax(bk, axis, do_causal_mask, name, thread_count);
}

} // namespace mllm::xnnpack