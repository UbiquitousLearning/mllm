#include "backends/xnnpack/Ops/XpLinear.hpp"
#include "xnnpack.h"
#include <cassert>
#include <limits>

namespace mllm::xnnpack {

XpLinear::XpLinear(Backend *bk, const std::string &op_name, int in_features, int out_features, bool bias, int thread_count) :
    Op(bk, op_name), in_features_(in_features), out_features_(out_features), bias_(bias), thread_count_(thread_count) {
    weight_params_.setBackend(bk);
    bias_params_.setBackend(bk);
}

ErrorCode XpLinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // do nothing
    return MLLM_NO_ERROR;
}

ErrorCode XpLinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs[0]->dimension() == in_features_);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    return Op::reshape(inputs, outputs);
}

ErrorCode XpLinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto xpb = (XnnpackBackend *)backend();
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), inputs);
    tryDefineAllXpTensors(xpb->getCurProcessingGraph(), outputs);

    if (xpb->getCurProcessingGraph()->getExecCnt()) return MLLM_NO_ERROR;

    defineWeightTensor(xpb->getCurProcessingGraph(), &weight_params_, {(size_t)out_features_, (size_t)in_features_});
    if (bias_) {
        defineWeightTensor(xpb->getCurProcessingGraph(), &bias_params_, {(size_t)out_features_});
    }

    // FIXME: output_min and output_max should be judged based on outputs' dtype
    auto status = xnn_define_fully_connected(
        xpb->getCurProcessingGraph()->getXnnSubgraph(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::max(),
        inputs[0]->uuid(),
        weight_params_.uuid(),
        bias_ ? bias_params_.uuid() : XNN_INVALID_VALUE_ID,
        outputs[0]->uuid(),
        0);

    if (status != xnn_status_success) {
        Log::error("XpLinear::execute Error");
        exit(-1);
    }
    return MLLM_NO_ERROR;
}

ErrorCode XpLinear::load(AbstructLoader &loader) {
    auto xpb = (XnnpackBackend *)backend();

    weight_params_.setName(name() + ".weight");
    weight_params_.reshape(1, 1, out_features_, in_features_);
    if (loader.getDataType(weight_params_.name()) != MLLM_TYPE_COUNT) {
        weight_params_.setDtype(loader.getDataType(weight_params_.name()));
        weight_params_.alloc();
        loader.load(&weight_params_);
    } else {
        weight_params_.setDtype(Op::noLoadWeightsDtype());
        weight_params_.alloc();
    }

    if (bias_) {
        bias_params_.setName(name() + ".bias");
        bias_params_.reshape(1, 1, 1, out_features_);
        if (loader.getDataType(bias_params_.name()) != MLLM_TYPE_COUNT) {
            bias_params_.setDtype(loader.getDataType(bias_params_.name()));
            bias_params_.alloc();
            loader.load(&bias_params_);
        } else {
            bias_params_.setDtype(Op::noLoadWeightsDtype());
            bias_params_.alloc();
        }
    }
    return Op::load(loader);
}

ErrorCode XpLinear::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_params_.free();
    if (bias_) {
        bias_params_.free();
    }
    return Op::free(inputs, outputs);
}

Op *XpLinearCreator::create(OpParam op_param, Backend *bk, const string &name, int thread_count) const {
    int in_features = static_cast<int>(op_param["in_features"]);
    int out_features = static_cast<int>(op_param["out_features"]);
    bool bias = static_cast<bool>(op_param["bias"]);
    return new XpLinear(bk, name, in_features, out_features, bias, thread_count);
}

} // namespace mllm::xnnpack