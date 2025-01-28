#include "backends/xnnpack/Ops/XpParameter.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "xnnpack.h"

namespace mllm::xnnpack {

XpParameter::XpParameter(Backend *bn, const string &op_name, int batch, int head, int seq, int dim, int thread_count) :
    thread_count(thread_count),
    Op(bn, op_name) {
    batch_ = batch;
    head_ = head;
    seq_ = seq;
    dim_ = dim;
    weight_.setBackend(bn);
}

ErrorCode XpParameter::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(batch_, head_, seq_, dim_);
    return Op::reshape(inputs, outputs);
}

ErrorCode XpParameter::load(AbstructLoader &loader) {
    weight_.setName(name());
    weight_.reshape(batch_, head_, seq_, dim_);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    return Op::load(loader);
}

ErrorCode XpParameter::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    Log::warn("XpParameter::execute will copy weight's value, may have perfromance issues");
    auto xpb = (XnnpackBackend *)backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);
    defineWeightTensor(xpb->getCurProcessingGraph(), &weight_);

    // copy
    auto status = xnn_define_copy(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        weight_.uuid(),
        outputs[0]->uuid(),
        0);
    if (status != xnn_status_success) {
        Log::error("XpParameter::execute xnn_define_copy failed");
        exit(-1);
    }

    return Op::execute(inputs, outputs);
}

ErrorCode XpParameter::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}

ErrorCode XpParameter::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->shallowCopyFrom(&weight_, false);
    return MLLM_NO_ERROR;
}

Op *XpParameterCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    int batch = (int)op_param["batch"];
    int head = (int)op_param["head"];
    int seq = (int)op_param["seq"];
    int dim = (int)op_param["dim"];
    Log::error("XpParameterCreator::create, Xnnpack backend not support XpParameter yet");
    exit(-1);
    return new XpParameter(bk, name, batch, head, seq, dim, thread_count);
}
} // namespace mllm::xnnpack