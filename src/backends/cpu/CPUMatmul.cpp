
#include "CPUMatmul.hpp"

namespace mllm {

// template class CPUMatmul;
// template class CPUMatmul;

CPUMatmul::CPUMatmul(Backend *bn, bool transposeA, bool transposeB, bool transposeC, bool multiThread) :
    Op(bn) {
}

ErrorCode CPUMatmul::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUMatmul  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->channels(), inputs[1]->num());
    CHECK_EQ(inputs[0]->height(), inputs[1]->height());
    CHECK_EQ(inputs[0]->width(), inputs[1]->width());
    outputs[0]->reshape(inputs[0]->num(), inputs[1]->channels(), inputs[0]->height(), inputs[0]->width());
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
    int H = inputs[0]->height();
    int W = inputs[0]->width();
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    float value = 0;
                    for (int k = 0; k < K; k++) {
                        value += inputs[0]->dataAt<float>(m, k, h, w) * inputs[1]->dataAt<float>(k, n,  h, w);
                    }
                    outputs[0]->setDataAt<float>(m, n,  h, w, value);
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
