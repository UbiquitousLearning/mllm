#include "QNNCommonOp.hpp"
#include "QnnWrapperUtils.hpp"
#include "Types.hpp"

namespace mllm {

QNNCommonOp::QNNCommonOp(Backend *bn, string opName) :
    Op(bn, opName) {
    qnnBackend_ = dynamic_cast<QNNBackend *>(bn);
}

ErrorCode QNNCommonOp::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#if DEBUG
    std::cout << "*QNN" << name_ << " reshape*" << std::endl;
#endif
    return NO_ERROR;
}

ErrorCode QNNCommonOp::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#if DEBUG
    std::cout << "*QNN" << name_ << " execute*" << std::endl;
#endif
    return NO_ERROR;
}

ErrorCode QNNCommonOp::load(AbstructLoader &loader) {
#if DEBUG
    std::cout << "*QNN" << name_ << " *" << std::endl;
#endif
    return NO_ERROR;
}

ErrorCode QNNCommonOp::graphAddNode(string name, string nodeType, vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, string packageName) {
    if (qnn_wrapper_api::ModelError_t::MODEL_NO_ERROR != qnnBackend_->graphAddNode(name, nodeType, inputs, outputs)) {
        return ErrorCode::INVALID_VALUE;
    }
    return NO_ERROR;
}

} // namespace mllm
