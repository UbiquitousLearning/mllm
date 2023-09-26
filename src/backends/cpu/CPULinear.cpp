
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
    if(!inputs[0]->allocted()){
        inputs[0]->alloc(); //TODO remove
    }
    outputs[0]->alloc();
    weight_.alloc();
    return NO_ERROR;
}

ErrorCode CPULinear::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPULinear()" << std::endl;
    // INPUT: M.K
    // W:K,N
    // OUTPUT:M.N
    int M = inputs[0]->height();
    int K = in_features_;
    int N = out_features_;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            outputs[0]->setDataAt<float>(1, 1, m, n, 0.0);
            for (int k = 0; k < K; k++) {
                auto mm_v = inputs[0]->dataAt<float>(1, 1, m, k) * weight_.dataAt<float>(1, 1, k, n);
                outputs[0]->setDataAt<float>(1, 1, m, n, outputs[0]->dataAt<float>(1, 1, m, n) + mm_v);
            }
        }
    }

    return NO_ERROR;
}

ErrorCode CPULinear::load(ParamLoader &loader) {
    std::cout << "CPULinear load" << std::endl;
    loader.load(&weight_);
    return NO_ERROR;
}

} // namespace mllm
