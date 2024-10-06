#include "backends/xnnpack/Ops/XpDispatch.hpp"
#include "Types.hpp"

namespace mllm::xnnpack {

ErrorCode XpDispatch::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do nothing
    return ErrorCode::MLLM_NO_ERROR;
}

ErrorCode XpDispatch::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xnnbk = (XnnpackBackend *)(this->backend());

    // recreate runtime
    auto m_rt = xnnbk->recreateModelRuntime(thread_count_);

    // create Model
    m_rt->createModel(xnnbk->getXnnSubgraph());

    // create runtime
    m_rt->createRuntime(0);

    // reshape
    m_rt->reshapeRuntime();

    // setup
    m_rt->setupRuntime();

    // run
    if (!m_rt->invoke()) {
        return ErrorCode::NO_EXECUTION;
    }

    return ErrorCode::MLLM_NO_ERROR;
}

ErrorCode XpDispatch::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do nothing
    return ErrorCode::MLLM_NO_ERROR;
}

Op *XpDispatchCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpDispatch(bk, name, thread_count);
}

} // namespace mllm::xnnpack