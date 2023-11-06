
#include "CPUSoftMax.hpp"
#include <cmath>
namespace mllm {

// template class CPUSoftMax;
// template class CPUSoftMax;

CPUSoftMax::CPUSoftMax(Backend *bn, string opName, int axis, bool multiThread) :
    Op(bn, opName) {
    axis_ = axis;
}

ErrorCode CPUSoftMax::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPUSoftMax  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    //outputs[0]->setDtype(activationDtype());
    return NO_ERROR;
}

ErrorCode CPUSoftMax::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout << name() << "  CPUSoftMax()" << std::endl;
    auto &input = inputs[0];
    auto &output = outputs[0];

    float max_ = -9999;
    // 计算最大值
    for (int n = 0; n < input->shape(0); ++n) {
        for (int c = 0; c < input->shape(1); ++c) {
            for (int h = 0; h < input->shape(2); ++h) {
                for (int w = 0; w < input->shape(3); ++w) {
                    std::vector<int> index = {n, c, h, w};
                    if (input->dataAt<float>(index) > max_) {
                        max_ = input->dataAt<float>(index);
                    }
                }
            }
        }
    }

    for (int n = 0; n < input->shape(0); ++n) {
        for (int c = 0; c < input->shape(1); ++c) {
            for (int h = 0; h < input->shape(2); ++h) {
                for (int w = 0; w < input->shape(3); ++w) {
                    std::vector<int> index = {n, c, h, w};
                    int num_classes = input->shape(axis_); // 获取类别数量
                    // 计算指定类别的 softmax
                    float sum_exp = 0.0;
                    for (int j = 0; j < num_classes; j++) {
                        index[axis_] = j;
                        sum_exp += std::exp(input->dataAt<float>(index) - max_);
                    }
                    // 将 softmax 结果写入输出Tensor
                    for (int j = 0; j < num_classes; j++) {
                        index[axis_] = j;
                        float softmax_value = std::exp(input->dataAt<float>(index)- max_) / sum_exp;
                        output->setDataAt<float>(index, softmax_value);
                    }
                }
            }
        }
    }

#ifdef DEBUG
    inputs[0]->printData<float>();
    outputs[0]->printData<float>();
#endif
    return NO_ERROR;
}

ErrorCode CPUSoftMax::load(ParamLoader &loader) {
    std::cout << name() << "  CPUSoftMax load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
