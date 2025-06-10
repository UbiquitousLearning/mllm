
#include "QNNSubGraphFinalize.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <memory>

namespace mllm {
QNNSubGraphFinalize::QNNSubGraphFinalize(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNSubGraphFinalize::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for(auto& t : inputs) {
        t->setTtype(GRAPH_OUTPUT);
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSubGraphFinalize::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for (auto input : inputs) {
        input->to(MLLM_CPU);
    }

    this->backend_->onSetUpEnd(inputs, outputs);
    return MLLM_NO_ERROR;
}

ErrorCode QNNSubGraphFinalize::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

} // namespace mllm
