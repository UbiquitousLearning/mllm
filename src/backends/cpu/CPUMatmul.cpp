
#include "CPUMatmul.hpp"

namespace mllm {

CPUMatmul::CPUMatmul(Backend *bn, bool transpose0, bool transpose1, bool multiThread) :
    Op(bn) {
    transpose0_ = transpose0;
    transpose1_ = transpose1;
    support_multi_thread_ = multiThread;
}

ErrorCode CPUMatmul::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUMatmul  reshape" << std::endl;
    if (!transpose0_ && !transpose1_) {
        CHECK_EQ(inputs.size(), 2);
        CHECK_EQ(outputs.size(), 1);
        CHECK_EQ(inputs[0]->width(), inputs[1]->width());
        CHECK_EQ(inputs[0]->width(), 1);
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->num(), inputs[1]->num());
        CHECK_EQ(inputs[0]->height(), inputs[1]->channels());
        outputs[0]->reshape(inputs[0]->num(), inputs[0]->channels(), inputs[1]->height(), inputs[0]->width());
    }
    return NO_ERROR;
}

ErrorCode CPUMatmul::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUMatmul  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    if (!inputs[1]->allocted()) {
        inputs[1]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    return NO_ERROR;
}

ErrorCode CPUMatmul::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUMatmul()" << std::endl;
    // INPUT: M.K
    // W:K,N
    // OUTPUT:M.N
    int M = inputs[0]->num();
    int K = inputs[0]->channels();
    int N = inputs[1]->channels();
    for (int b = 0; b < inputs[0]->num(); b++) {
        for (int w = 0; w < inputs[0]->width(); w++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float value = 0;
                    for (int k = 0; k < K; k++) {
                        value += inputs[0]->dataAt<float>(b, m, k, w) * inputs[1]->dataAt<float>(b, k, n, w);
                    }
                    outputs[0]->setDataAt<float>(b, m, n, w, value);
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode CPUMatmul::load(ParamLoader &loader) {
    std::cout << "CPUMatmul load" << std::endl;
    return NO_ERROR;
}

} // namespace mllm
