#include "backends/xnnpack/Ops/XpD2H.hpp"
#include "Types.hpp"
#include "xnnpack.h"
#include "xnnpack/XnnpackBackend.hpp"

namespace mllm::xnnpack {

ErrorCode XpD2H::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode XpD2H::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto i = inputs[0];
    auto o = outputs[0];
    o->reshape(i->batch(), i->head(), i->sequence(), i->dimension());
    o->uuid() = i->uuid();
    return Op::reshape(inputs, outputs);
}

ErrorCode XpD2H::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto o = outputs[0];
    auto i = inputs[0];
    o->uuid() = inputs[0]->uuid();
    o->forceResetHostPointer(((XnnpackBackend *)backend())->getCurProcessingGraph()->getExternalValueptr(o->uuid()));
    o->uuid() = XNN_INVALID_VALUE_ID;
    i->forceResetHostPointer(((XnnpackBackend *)backend())->getCurProcessingGraph()->getExternalValueptr(i->uuid()));
    i->uuid() = XNN_INVALID_VALUE_ID;
    return MLLM_NO_ERROR;
}

Op *XpD2HCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpD2H(bk, name, thread_count);
}
} // namespace mllm::xnnpack