#include "NNAPIScale.hpp"
#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"
#include "Types.hpp"
#include <cstdint>

namespace mllm {

NNAPIScale::NNAPIScale(Backend *bn, string opName, float scale, float bias, bool bias_after_scale) :
    NNAPICommonOp(bn, opName) {
    scale_ = scale;
    bias_ = bias;
    bias_after_scale_ = bias_after_scale;
}

ErrorCode NNAPIScale::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " reshape*" << std::endl;
#endif
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(),
                        inputs[0]->head(),
                        inputs[0]->sequence(),
                        inputs[0]->dimension());
    return NO_ERROR;
}

ErrorCode NNAPIScale::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
#ifdef DEBUG
    std::cout << "*NNAPI " << name() << " setUp*" << std::endl;
#endif
    outputs[0]->alloc();

    // uint32_t channel = scaleParam->channels();
    auto inputShape = inputs[0]->shape(); // NCHW
    int inputSize = inputs[0]->count();   // element size

    std::vector<uint32_t> dims = {1, 1, 1, 1};

    auto result = NO_ERROR;
    if (bias_after_scale_) { // output = middle + bias
        auto middleIdx = buildTensor(ANEURALNETWORKS_TENSOR_FLOAT32, inputShape);
        auto inputIdxs = getTensorIdxs(inputs);
        inputIdxs.push_back(buildConstant(&scale_, sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, dims));
        inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));
        buildOperation(ANEURALNETWORKS_MUL, inputIdxs, {middleIdx});

        auto biasIdx = buildConstant(&bias_, sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, dims);
        result = buildOperation(ANEURALNETWORKS_ADD,
                                {middleIdx, biasIdx, buildScalar(ANEURALNETWORKS_FUSED_NONE)},
                                getTensorIdxs(outputs));
    } else { // output = middle * scale
        auto middleIdx = buildTensor(ANEURALNETWORKS_TENSOR_FLOAT32, inputShape);
        auto inputIdxs = getTensorIdxs(inputs);
        inputIdxs.push_back(buildConstant(&bias_, sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, dims));
        inputIdxs.push_back(buildScalar(ANEURALNETWORKS_FUSED_NONE));
        buildOperation(ANEURALNETWORKS_ADD, inputIdxs, {middleIdx});

        auto constantIdx = buildConstant(&scale_, sizeof(float), ANEURALNETWORKS_TENSOR_FLOAT32, dims);
        result = buildOperation(ANEURALNETWORKS_MUL,
                                {middleIdx, constantIdx, buildScalar(ANEURALNETWORKS_FUSED_NONE)},
                                getTensorIdxs(outputs));
    }
    return result;
}
} // namespace mllm
