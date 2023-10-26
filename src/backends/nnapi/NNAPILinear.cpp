#include "NNAPILinear.hpp"
#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"
#include "Types.hpp"

namespace mllm {

NNAPILinear::NNAPILinear(Backend *bn, int in_features, int out_features, bool bias) :
    NNAPICommonOp(bn) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode NNAPILinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPILinear reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    // N     |    C       |   H                   |  W
    // -----------------------------------------------
    // 1     |out_channel | in_channel            |  1
    //       |out_features| in_features           |
    // -----------------------------------------------
    // batch |in_channel  | seq_len               |  1
    //       |in_features | inputs[0]->sequence()   |
    // -----------------------------------------------
    // batch |out_channel | seq_len               |  1
    //       |out_features|  inputs[0]->sequence()  |
    CHECK_EQ(in_features_, inputs[0]->dimension());

    weight_.reshape(1, inputs[0]->head(), out_features_, in_features_);
    weight_.setName(name() + ".weight");
    weight_.setDtype(weightsDtype());
    if (support_bias_) {
        bias_.reshape(1, inputs[0]->head(), 1, out_features_);
        bias_.setName(name() + ".bias");
        bias_.setDtype(weightsDtype());
    }
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    outputs[0]->setDtype(activationDtype());
    return NO_ERROR;
}

ErrorCode NNAPILinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPILinear setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    weight_.alloc();
    //    weight_.fullData<float>(1);
    if (support_bias_) {
        bias_.alloc();
    }

    return buildOperation(ANEURALNETWORKS_FULLY_CONNECTED, getTensorIdxs(inputs), getTensorIdxs(outputs));
}

ErrorCode NNAPILinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << "NNAPILinear execute do nothing" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
