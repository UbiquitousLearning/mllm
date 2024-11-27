
#include "CPUConvolution3D.hpp"
#include "../compute/Convolution.hpp"

namespace mllm {

CPUConvolution3D::CPUConvolution3D(Backend *bn, string opName, int in_channel, int out_channel, vector<int> kernal_size, vector<int> stride, PaddingType padding_type, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    kernel_size_[0] = kernal_size[0];
    kernel_size_[1] = kernal_size[1];
    kernel_size_[2] = kernal_size[2];
    stride_[0] = stride[0];
    stride_[1] = stride[1];
    stride_[2] = stride[2];
    in_channel_ = in_channel;
    out_channel_ = out_channel;
    padding_type_ = padding_type;
    support_bias_ = bias;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode CPUConvolution3D::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // batch = batch
    // sequence = out_channel
    // head = height
    // dimension = width
    assert(inputs[0]->ctype() == BCTHW);
    assert(in_channel_ == inputs[0]->channel());
    switch (padding_type_) {
    case SAME: {
        padding_t_ = (kernel_size_[0] - 1) / 2;
        padding_h_ = (kernel_size_[1] - 1) / 2;
        padding_w_ = (kernel_size_[2] - 1) / 2;
        const int out_time = (inputs[0]->time() + 2 * padding_t_ - kernel_size_[0]) / stride_[0] + 1;
        const int out_height = (inputs[0]->height() + 2 * padding_h_ - kernel_size_[1]) / stride_[0] + 1;
        const int out_width = (inputs[0]->width() + 2 * padding_w_ - kernel_size_[2]) / stride_[1] + 1;
        outputs[0]->reshape(inputs[0]->batch(), out_channel_, out_time, out_height, out_width);
        break;
    }
    case VALID: {
        padding_t_ = 0;
        padding_h_ = 0;
        padding_w_ = 0;
        const int out_time = (inputs[0]->time() - kernel_size_[0]) / stride_[0] + 1;
        const int out_height = (inputs[0]->height() - kernel_size_[1]) / stride_[1] + 1;
        const int out_width = (inputs[0]->width() - kernel_size_[2]) / stride_[2] + 1;
        outputs[0]->reshape(inputs[0]->batch(), out_channel_, out_time, out_height, out_width);
        break;
    }
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUConvolution3D::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(out_channel_, in_channel_, kernel_size_[0], kernel_size_[1], kernel_size_[2]);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
        kernal_ = reshape_conv3d_kernal_fp32(&weight_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
        kernal_ = reshape_conv3d_kernal_fp32(&weight_);
    }
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, 1, out_channel_);
        if (loader.getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader.getDataType(bias_.name()));
            bias_.alloc();
            loader.load(&bias_);
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }
    return Op::load(loader);
}

ErrorCode CPUConvolution3D::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    switch (padding_type_) {
    case SAME: {
        // conv2d_fp32_SAME(inputs[0].get(), outputs[0].get(), &weight_, support_bias_, &bias_, stride_[0], stride_[1], padding_h_, padding_w_, thread_count);
        std::cout << "TO SUPPORT" << std::endl;
        break;
    }
    case VALID: {
        // conv3d_fp32_VALID(inputs[0].get(), outputs[0].get(), &weight_, support_bias_, &bias_,stride_[0], stride_[1], stride_[2], thread_count);
        conv3d_fp32_VALID(inputs[0].get(), outputs[0].get(), kernal_, kernel_size_[0], kernel_size_[1], kernel_size_[2], support_bias_, &bias_, stride_[0], stride_[1], stride_[2], thread_count);
        break;
    }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUConvolution3D::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}

ErrorCode CPUConvolution3D::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}
} // namespace mllm
