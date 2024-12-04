
#include "CPUMaxPool2D.hpp"
#include "../compute/Pooling.hpp"

namespace mllm {

CPUMaxPool2D::CPUMaxPool2D(Backend *bn, string opName, vector<int> kernal_size, vector<int> stride, PaddingType padding_type, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    kernel_size_[0] = kernal_size[0];
    kernel_size_[1] = kernal_size[1];
    stride_[0] = stride[0];
    stride_[1] = stride[1];
    padding_type_ = padding_type;
}

ErrorCode CPUMaxPool2D::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // batch = batch
    // sequence = out_channel
    // head = height
    // dimension = width
    switch (padding_type_) {
    case SAME: {
        padding_h_ = (kernel_size_[0] - 1) / 2;
        padding_w_ = (kernel_size_[1] - 1) / 2;
        const int out_height = (inputs[0]->head() + 2 * padding_h_ - kernel_size_[0]) / stride_[0] + 1;
        const int out_width = (inputs[0]->dimension() + 2 * padding_w_ - kernel_size_[1]) / stride_[1] + 1;
        outputs[0]->reshape(inputs[0]->batch(), out_height, inputs[0]->sequence(), out_width);
        break;
    }
    case VALID: {
        padding_h_ = 0;
        padding_w_ = 0;
        const int out_height = (inputs[0]->head() - kernel_size_[0]) / stride_[0] + 1;
        const int out_width = (inputs[0]->dimension() - kernel_size_[1]) / stride_[1] + 1;
        outputs[0]->reshape(inputs[0]->batch(), out_height, inputs[0]->sequence(), out_width);
        break;
    }
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMaxPool2D::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    switch (padding_type_) {
    case SAME: {
        maxpool2d_fp32_SAME(inputs[0].get(), outputs[0].get(), kernel_size_[0], kernel_size_[1], stride_[0], stride_[1], padding_h_, padding_w_, thread_count);
        break;
    }
    case VALID: {
        maxpool2d_fp32_VALID(inputs[0].get(), outputs[0].get(), kernel_size_[0], kernel_size_[1], stride_[0], stride_[1], thread_count);
        break;
    }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUMaxPool2D::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}
} // namespace mllm
