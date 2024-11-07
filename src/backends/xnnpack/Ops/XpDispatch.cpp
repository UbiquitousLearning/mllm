#include "backends/xnnpack/Ops/XpDispatch.hpp"
#include "Types.hpp"

namespace mllm::xnnpack {

ErrorCode XpDispatch::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do nothing
    return ErrorCode::MLLM_NO_ERROR;
}

ErrorCode XpDispatch::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xnnbk = (XnnpackBackend *)(this->backend());
    auto cargo = xnnbk->getCurProcessingGraph();

    // recreate runtime
    auto m_rt = cargo->recreateModelRuntime();

    // create Model
    m_rt->createModel(cargo->getXnnSubgraph());

    // create runtime
    m_rt->createRuntime(0);

    // auto wc = xnnbk->getWeightCache();
    // if (!xnnbk->isWeightCacheFinalized()) {
    //     xnn_finalize_weights_cache(wc, xnn_weights_cache_finalization_kind_hard);
    //     xnnbk->setWeightCacheFinalized(true);
    // }

    // reshape
    m_rt->reshapeRuntime();

    // setup
    m_rt->setupRuntime();

    // run
    if (!m_rt->invoke()) {
        return ErrorCode::NO_EXECUTION;
    }

    // update all output's ptr
    cargo->assignPtrToTensor();

    cargo->setSubgraphDispatched(true);

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