
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
    CHECK_EQ(inputs[0]->head(), inputs[1]->head());
    //    CHECK_EQ(inputs[0]->head(), 1);
    CHECK_EQ(inputs[0]->batch(), inputs[1]->batch());
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
        CHECK_EQ(inputs[0]->dimension(), inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->dimension());
    } else if (transpose1_) {
        /*
         N     |    C       |   H                   |  W
         -----------------------------------------------
         batch |in_channel | out_channel            |  1
         -----------------------------------------------
         batch |in_channel  | seq_len               |  1
         -----------------------------------------------
         batch |out_channel | seq_len               |  1
         */
        CHECK_EQ(inputs[0]->dimension(), inputs[1]->dimension());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[1]->sequence());
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
        CHECK_EQ(inputs[0]->sequence(), inputs[1]->sequence());
        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->dimension(), inputs[1]->dimension());
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
        //        M = inputs[0]->dimension();
        //        K = inputs[0]->sequence();
        //        N = inputs[1]->sequence();

        M = inputs[0]->sequence();
        K = inputs[0]->dimension();
        N = inputs[1]->dimension();
    } else if (transpose1_) {
        //        M = inputs[0]->sequence();
        //        K = inputs[0]->dimension();
        //        N = inputs[1]->sequence();

        M = inputs[0]->sequence();
        K = inputs[0]->dimension();
        N = inputs[1]->sequence();
    } else {
        //        M = inputs[0]->dimension();
        //        K = inputs[0]->sequence();
        //        N = inputs[1]->dimension();

        M = inputs[0]->dimension();
        K = inputs[0]->sequence();
        N = inputs[1]->dimension();
    }
    for (int b = 0; b < inputs[0]->batch(); b++) {
        for (int w = 0; w < inputs[0]->head(); w++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float value = 0;
                    for (int k = 0; k < K; k++) {
                        if (!transpose0_ && !transpose1_) {
                            value += inputs[0]->dataAt<float>(b, w, m, k) * inputs[1]->dataAt<float>(b, w, k, n);
                        } else if (transpose1_) {
                            value += inputs[0]->dataAt<float>(b, w, m, k) * inputs[1]->dataAt<float>(b, w, n, k);
                        } else {
                            value += inputs[0]->dataAt<float>(b, w, k, m) * inputs[1]->dataAt<float>(b, w, k, n);
                        }
                    }
                    outputs[0]->setDataAt<float>(b, w, m, n, value);
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
