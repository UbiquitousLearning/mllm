
#include "CPUSoftMax.hpp"

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
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < channels; ++c) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int i = 0; i < channels; ++i) {
                        max_val = std::max(max_val, input->dataAt<float>(n, i, h, w));
                    }

                    float sum_exp = 0.0;
                    for (int i = 0; i < channels; ++i) {
                        sum_exp += std::exp(input->dataAt<float>(n, i, h, w) - max_val);
                    }

                    float softmax_val = std::exp(input->dataAt<float>(n, c, h, w) - max_val) / sum_exp;
                    output->setDataAt<float>(n, c, h, w, softmax_val);
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
