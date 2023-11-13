#include <cmath>
#include "NNAPICommonOp.hpp"
#include "NNAPINeuralNetworks.h"
#include "NNAPIRMSNorm.hpp"
#include "Tensor.hpp"

namespace mllm {

// template class NNAPIRMSNorm;
// template class NNAPIRMSNorm;

NNAPIRMSNorm::NNAPIRMSNorm(Backend *bn, string opName, float epsilon) :
    NNAPICommonOp(bn, opName), epsilon_(epsilon) {
    weight_.setBackend(bn);
}

ErrorCode NNAPIRMSNorm::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // RMSNorm 类似于LayerNorm作用于channel维度
    weight_.reshape(1, 1, 1, inputs[0]->dimension()); // (C, 1, 1, 1)
    weight_.setName(name() + ".weight");
    weight_.setDtype(weightsDtype());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    outputs[0]->setDtype(activationDtype());
    std::cout << name() << "  NNAPIRMSNorm  reshape" << std::endl;
    return NO_ERROR;
}

ErrorCode NNAPIRMSNorm::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    weight_.alloc();

    std::cout << name() << "  NNAPIRMSNorm  setUp" << std::endl;

    return buildOperation(ANEURALNETWORKS_ADD, {}, {});
}

ErrorCode NNAPIRMSNorm::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  NNAPIRMSNorm() do nothing" << std::endl;
    return NO_ERROR;
}
ErrorCode NNAPIRMSNorm::load(ParamLoader &loader) {
    loader.load(&weight_);
    return Op::load(loader);
}
ErrorCode NNAPIRMSNorm::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm