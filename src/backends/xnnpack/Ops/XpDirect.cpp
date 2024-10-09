#include "backends/xnnpack/Ops/XpDirect.hpp"
#include "Types.hpp"
#include "xnnpack.h"
#include "backends/xnnpack/Utils/Logger.hpp"

namespace mllm::xnnpack {

ErrorCode XpDirect::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xnnbk = (XnnpackBackend *)(this->backend());

    switch (type_) {
    case XpTensorType::Normal: {
        for (auto i = 0; i < inputs.size(); ++i) {
            inputs[i]->uuid() = XNN_INVALID_VALUE_ID;
            outputs[i]->uuid() = XNN_INVALID_VALUE_ID;
        }

        Log::warn("XpDirect Op with XpTensorType::Normal will do nothing for you.");
        break;
    }
    case XpTensorType::ExternalInput: {
        for (auto i = 0; i < inputs.size(); ++i) {
            inputs[i]->uuid() = XNN_INVALID_VALUE_ID;

            defineXpTensor(xnnbk, inputs[i], type_);

            outputs[i]->uuid() = inputs[i]->uuid();
        }
        break;
    }
    case XpTensorType::ExternalOutput: {
        for (auto i = 0; i < inputs.size(); ++i) {
            inputs[i]->uuid() = XNN_INVALID_VALUE_ID;

            defineXpTensor(xnnbk, inputs[i], type_);

            outputs[i]->uuid() = inputs[i]->uuid();
        }
        break;
    }
    };

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
    return MLLM_NO_ERROR;
}

Op *XpDirectCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    return new XpDirect(bk, name, thread_count);
}
} // namespace mllm::xnnpack