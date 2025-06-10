
#include "QNNSubGraphStart.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <memory>

namespace mllm {
QNNSubGraphStart::QNNSubGraphStart(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNSubGraphStart::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSubGraphStart::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for(auto input : inputs) {
        input->to(MLLM_QNN);
        input->alloc();
    }

    this->backend_->onSetUpStart(inputs, outputs, name_);
    return MLLM_NO_ERROR;
}

ErrorCode QNNSubGraphStart::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return MLLM_NO_ERROR;
}

ErrorCode QNNSubGraphStart::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    this->backend_->onExecuteStart(inputs, outputs, name_);
    return MLLM_NO_ERROR;
}



} // namespace mllm
