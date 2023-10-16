
#include "CPULinear.hpp"

namespace mllm {

CPULinear::CPULinear(Backend *bn, string opName, int in_features, int out_features, bool bias, bool multiThread) :
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    support_multi_thread_ = multiThread;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode CPULinear::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPULinear  reshape" << std::endl;
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
    weight_.reshape(1, inputs[0]->head(), in_features_, out_features_);
    weight_.setName(name() + ".weight");
    bias_.reshape(1, inputs[0]->head(), 1, out_features_);
    bias_.setName(name() + ".bias");
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    return NO_ERROR;
}

ErrorCode CPULinear::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPULinear  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    weight_.alloc();
    //    weight_.fullData<float>(1);
    bias_.alloc();
    return NO_ERROR;
}

ErrorCode CPULinear::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPULinear()" << std::endl;
    // INPUT: M.K
    // W:K,N
    // OUTPUT:M.N
    int M = inputs[0]->sequence();
    int K = in_features_;  // inputs[0]->dimension()
    int N = out_features_; // inputs[1]->dimension()
    for (int b = 0; b < inputs[0]->batch(); b++) {
        for (int h = 0; h < inputs[0]->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float value = 0;
                    for (int k = 0; k < K; k++) {
                        value += inputs[0]->dataAt<float>(0, h, m, k) * weight_.dataAt<float>(b, h, k, n);
                    }
                    if (support_bias_) {
                        value += bias_.dataAt<float>(0, h, 0, n);
                    }
                    outputs[0]->setDataAt<float>(b, h, m, n, value);
                }
            }
        }
    }

#ifdef DEBUG
    inputs[0]->printData<float>();
    weight_.printData<float>();
    bias_.printData<float>();
    outputs[0]->printData<float>();
#endif

    return NO_ERROR;
}

ErrorCode CPULinear::load(ParamLoader &loader) {
    std::cout << name() << "  CPULinear load" << std::endl;
    loader.load(&weight_);
    if (support_bias_)
        loader.load(&bias_);
    return NO_ERROR;
}
ErrorCode CPULinear::reshapeOutputs(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPULinear  reshape" << std::endl;
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
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    outputs[0]->alloc();
    return NO_ERROR;
}

} // namespace mllm
