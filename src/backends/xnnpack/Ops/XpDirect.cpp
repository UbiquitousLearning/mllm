
#include "Types.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include "backends/xnnpack/Ops/XpDirect.hpp"
#include "xnnpack/XpInterface.hpp"

namespace mllm::xnnpack {

ErrorCode XpDirect::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xnnbk = (XnnpackBackend *)(this->backend());
    for (int i = 0; i < inputs.size(); ++i) {
        switch (type_) {
        case XpTensorType::ExternalInput:
            inputs[i]->xnnTensorType() = TensorType::INPUT_TENSOR;
            outputs[i]->xnnTensorType() = TensorType::INPUT_TENSOR;
            break;
        case XpTensorType::ExternalOutput:
            inputs[i]->xnnTensorType() = TensorType::OUTPUT_TENSOR;
            outputs[i]->xnnTensorType() = TensorType::OUTPUT_TENSOR;
            break;
        default:
            inputs[i]->xnnTensorType() = TensorType::NORMAL_TENSOR;
            outputs[i]->xnnTensorType() = TensorType::NORMAL_TENSOR;
            break;
        }
    }
    return MLLM_NO_ERROR;
}

ErrorCode XpDirect::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs.size() != outputs.size()) {
        Log::error("XpDirect.reshape, inputs.size() != outputs.size(), {} != {}", inputs.size(), outputs.size());
    }

    for (auto i = 0; i < inputs.size(); ++i) {
        auto x = inputs[i];

        // the xnnpack only need mllm sides' dtype and shape
        outputs[i]->reshape(x->batch(), x->head(), x->sequence(), x->dimension());
        outputs[i]->setDtype(x->dtype());
    }

    return Op::reshape(inputs, outputs);
}

ErrorCode XpDirect::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xnnbk = (XnnpackBackend *)(this->backend());

    // define and register in xnnpack
    for (auto i = 0; i < inputs.size(); ++i) {
        auto in = inputs[i];
        auto out = outputs[i];

        defineXpTensor(xnnbk->getCurProcessingGraph(), in.get(), type_);

        out->uuid() = in->uuid();
        out->forceResetHostPointer(in->rawHostPtr());
    }
    return MLLM_NO_ERROR;
}

void XpDirect::setType(XpTensorType type) {
    type_ = type;
}

Op *XpDirectCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    auto ret = new XpDirect(bk, name, thread_count);
    ret->setType((XpTensorType)op_param["DirectType"]);
    return ret;
}
} // namespace mllm::xnnpack