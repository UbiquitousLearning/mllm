#include "backends/xnnpack/Ops/XpSubGraphFinalize.hpp"
#include "Types.hpp"

namespace mllm::xnnpack {

ErrorCode XpSubGraphFinalize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode XpSubGraphFinalize::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)(this->backend());
    xpb->getCurProcessingGraph()->recreateSubgraph();
    return MLLM_NO_ERROR;
}

ErrorCode XpSubGraphFinalize::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return ErrorCode::MLLM_NO_ERROR;
}

ErrorCode XpSubGraphFinalize::load(AbstructLoader &loader) {
    return ErrorCode::MLLM_NO_ERROR;
}

Op *XpSubGraphFinalizeCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpSubGraphFinalize(bk, name, thread_count);
}
} // namespace mllm::xnnpack