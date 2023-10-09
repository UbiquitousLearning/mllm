
#include "CPUAdd.hpp"

namespace mllm {

// template class CPUAdd;
// template class CPUAdd;

CPUAdd::CPUAdd(Backend *bn, bool multiThread) :
    Op(bn) {
}

ErrorCode CPUAdd::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUAdd  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(outputs.size(), 1);
    CHECK_EQ(inputs[0]->shape(0), inputs[1]->shape(0));
    CHECK_EQ(inputs[0]->shape(1), inputs[1]->shape(1));
    CHECK_EQ(inputs[0]->shape(2), inputs[1]->shape(2));
    CHECK_EQ(inputs[0]->shape(3), inputs[1]->shape(3));
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    return NO_ERROR;
}

ErrorCode CPUAdd::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUAdd  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    if (!inputs[1]->allocted()) {
        inputs[1]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    return NO_ERROR;
}

ErrorCode CPUAdd::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUAdd()" << std::endl;
    int N = inputs[0]->shape(0);
    int C = inputs[0]->shape(1);
    int H = inputs[0]->shape(2);
    int W = inputs[0]->shape(3);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    outputs[0]->setDataAt<float>(n, c, h, w, inputs[0]->dataAt<float>(n, c, h, w) + inputs[1]->dataAt<float>(n, c, h, w));
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode CPUAdd::load(ParamLoader &loader) {
    std::cout << "CPUAdd load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
