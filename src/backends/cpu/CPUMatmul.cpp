
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
    CHECK_EQ(inputs[0]->channels(), 1);
    CHECK_EQ(inputs[0]->width(), inputs[1]->height());
    outputs[0]->reshape(inputs[0]->num(), 1, inputs[0]->height(), inputs[1]->width());
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
    int M = inputs[0]->height();
    int K = inputs[0]->width();
    int N = inputs[1]->width();
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            outputs[0]->setDataAt<float>(1, 1, m, n, 0.0);
            for (int k = 0; k < K; k++) {
                auto mm_v = inputs[0]->dataAt<float>(1, 1, m, k) * inputs[1]->dataAt<float>(1, 1, k, n);
                outputs[0]->setDataAt<float>(1, 1, m, n, outputs[0]->dataAt<float>(1, 1, m, n) + mm_v);
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
