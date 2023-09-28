
#include "CPUSoftMax.hpp"
#include <cmath>
namespace mllm {

// template class CPUSoftMax;
// template class CPUSoftMax;

CPUSoftMax::CPUSoftMax(Backend *bn, int axis, bool multiThread) :
    Op(bn) {
    axis_ = axis;
}

ErrorCode CPUSoftMax::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSoftMax  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    outputs[0]->reshape(inputs[0]->num(), inputs[0]->channels(), inputs[0]->height(), inputs[0]->width());
    return NO_ERROR;
}

ErrorCode CPUSoftMax::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSoftMax  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    return NO_ERROR;
}

ErrorCode CPUSoftMax::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUSoftMax()" << std::endl;
    auto &input = inputs[0];
    auto &output = outputs[0];
    int num = input->num();
    int channels = input->channels();
    int height = input->height();
    int width = input->width();
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    std::vector<int> index = {n, c, h, w};
                    int num_classes = input->shape(axis_); // 获取类别数量
                    // 计算指定类别的 softmax
                    float sum_exp = 0.0;
                    for (int j = 0; j < num_classes; j++) {
                        index[axis_] = j;
                        sum_exp += std::exp(input->dataAt<float>(index));
                    }
                    // 将 softmax 结果写入输出Tensor
                    for (int j = 0; j < num_classes; j++) {
                        index[axis_] = j;
                        float softmax_value = std::exp(input->dataAt<float>(index)) / sum_exp;
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
    std::cout << "CPUSoftMax load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
