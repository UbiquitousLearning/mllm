
#include "CPULinear.hpp"

namespace mllm {

CPULinear::CPULinear(Backend *bn, int in_features, int out_features, bool bias, bool multiThread) :
    Op(bn) {
    in_features_ = in_features;
    out_features_ = out_features;
    bias_ = bias;
    support_multi_thread_ = multiThread;
    weight_.setBackend(bn);
}

ErrorCode CPULinear::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->channels(), 1);
    CHECK_EQ(inputs[0]->width(), in_features_);
    weight_.reshape(inputs[0]->num(), 1, in_features_, out_features_);
    outputs[0]->reshape(inputs[0]->num(), 1, inputs[0]->height(), out_features_);
    return NO_ERROR;
}

ErrorCode CPULinear::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear  setUp" << std::endl;
    outputs[0]->alloc();
    weight_.alloc();
    return NO_ERROR;
}

ErrorCode CPULinear::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear()" << std::endl;
    return NO_ERROR;
}

ErrorCode CPULinear::load(ParamLoader &loader) {
    std::cout << "CPULinear load" << std::endl;
    loader.load(&weight_);
    return NO_ERROR;
}

} // namespace mllm
