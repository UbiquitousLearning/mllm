
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
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->width(), inputs[1]->width());
    CHECK_EQ(inputs[0]->width(), 1);
    CHECK_EQ(inputs[0]->num(), inputs[1]->num());
    if (!transpose0_ && !transpose1_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->height(), inputs[1]->channels());
        outputs[0]->reshape(inputs[0]->num(), inputs[0]->channels(), inputs[1]->height(), inputs[0]->width());
    } else if (transpose0_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |in_channel | out_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->channels(), inputs[1]->channels());
        outputs[0]->reshape(inputs[0]->num(), inputs[0]->height(), inputs[1]->height(), inputs[0]->width());
    } else {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |out_channel | in_channel            |  1
         -----------------------------------------------
         batch |seq_len     | in_channel            |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->height(), inputs[1]->height());
        outputs[0]->reshape(inputs[0]->num(), inputs[0]->channels(), inputs[1]->channels(), inputs[0]->width());
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
    int M = 0;
    int K = 0;
    int N = 0;
    if (!transpose0_ && !transpose1_) {
        M = inputs[0]->channels();
        K = inputs[0]->height();
        N = inputs[1]->height();
    } else if (transpose0_){
        M = inputs[0]->height();
        K = inputs[0]->channels();
        N = inputs[1]->height();
    } else {
        M = inputs[0]->channels();
        K = inputs[0]->height();
        N = inputs[1]->channels();
    }
    for (int b = 0; b < inputs[0]->num(); b++) {
        for (int w = 0; w < inputs[0]->width(); w++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float value = 0;
                    for (int k = 0; k < K; k++) {
                        if (!transpose0_ && !transpose1_) {
                            value += inputs[0]->dataAt<float>(b, m, k, w) * inputs[1]->dataAt<float>(b, k, n, w);

                        } else if (transpose0_) {
                            value += inputs[0]->dataAt<float>(b, k, m, w) * inputs[1]->dataAt<float>(b, k, n, w);

                        } else {
                            value += inputs[0]->dataAt<float>(b, m, k, w) * inputs[1]->dataAt<float>(b, n, k, w);}
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
