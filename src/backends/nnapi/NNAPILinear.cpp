#include "NNAPILinear.hpp"
#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"
#include "Tensor.hpp"
#include "Types.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace mllm {

NNAPILinear::NNAPILinear(Backend *bn, string opName, int in_features, int out_features, bool bias) :
    NNAPICommonOp(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode NNAPILinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " reshape*" << std::endl;
#endif
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    // N     |    C       |   H                   |  W
    // -----------------------------------------------
    // 1     |out_channel | in_channel            |  1
    //       |out_features| in_features           |
    // -----------------------------------------------
    // batch |in_channel  | seq_len               |  1
    //       |in_features | inputs[0]->sequence()  |
    // -----------------------------------------------
    // batch |out_channel | seq_len               |  1
    //       |out_features|  inputs[0]->sequence()  |
    CHECK_EQ(in_features_, inputs[0]->dimension());
    // we assume that the input is 2D
    CHECK_EQ(1, inputs[0]->head());

    weight_.reshape(1, inputs[0]->head(), out_features_, in_features_);
    weight_.setName(name() + ".weight");
    weight_.setDtype(weightsDtype());

    // bias should be allocated even if it is not supported
    bias_.reshape(1, inputs[0]->head(), 1, out_features_);
    bias_.setName(name() + ".bias");
    bias_.setDtype(weightsDtype());

    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    outputs[0]->setDtype(activationDtype());
    return NO_ERROR;
}

ErrorCode NNAPILinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " setUp*" << std::endl;
#endif
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    weight_.alloc();
    bias_.alloc();
    if (!support_bias_) {
        bias_.fullData(0);
    }

    unsigned int batch_size = inputs[0]->batch() * inputs[0]->head() * inputs[0]->sequence();
    unsigned int num_units = out_features_;
    unsigned int input_size = in_features_;

    std::vector<uint32_t> inputIdxs;
    inputIdxs.push_back(getTensorIdx(inputs[0].get(), true, {batch_size, input_size})); // reshape to 2D
    inputIdxs.push_back(getTensorIdx(&weight_, true, {num_units, input_size}));
    inputIdxs.push_back(getTensorIdx(&bias_, true, {num_units}));
    inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));

    return buildOperation(ANEURALNETWORKS_FULLY_CONNECTED, inputIdxs, {getTensorIdx(outputs[0].get(), true, {batch_size, num_units})});
}

ErrorCode NNAPILinear::load(AbstructLoader &loader) {
#ifdef DEBUG
    std::cout << "*" << name() << " load*" << std::endl;
#endif
    loader.load(&weight_);
    if (support_bias_)
        loader.load(&bias_);
    return NO_ERROR;
}

} // namespace mllm
