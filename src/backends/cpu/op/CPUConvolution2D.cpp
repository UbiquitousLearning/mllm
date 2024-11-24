
#include "CPUConvolution2D.hpp"
#include "../compute/Convolution.hpp"

#include "../compute/Matmul.hpp"
#include "../compute/Im2Col.hpp"

namespace mllm {

CPUConvolution2D::CPUConvolution2D(Backend *bn, string opName, int in_channel, int out_channel, vector<int> kernal_size, vector<int> stride, PaddingType padding_type, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    kernel_size_[0] = kernal_size[0];
    kernel_size_[1] = kernal_size[1];
    stride_[0] = stride[0];
    stride_[1] = stride[1];
    in_channel_ = in_channel;
    out_channel_ = out_channel;
    padding_type_ = padding_type;
    support_bias_ = bias;
    weight_.setBackend(bn);
    bias_.setBackend(bn);

#ifdef __ARM_NEON
    im2col_layout_.setBackend(bn);
    output_not_transposed_.setBackend(bn);
#endif //! __ARM_NEON
}

ErrorCode CPUConvolution2D::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // batch = batch
    // sequence = out_channel
    // head = height
    // dimension = width
    assert(in_channel_ == inputs[0]->sequence());

    // #ifdef __ARM_NEON
    //     if (kernel_size_[0] == 16 && kernel_size_[1] == 16 && padding_h_ == 0 && padding_w_ == 0 && stride_[0] == 16 && stride_[1] == 16) {
    //         im2col_layout_.setDtype(inputs[0]->dtype());
    //         im2col_layout_.reshape(inputs[0]->batch(), 1, (inputs[0]->head() / 16) * (inputs[0]->dimension() / 16), 16 * 16 * in_channel_);
    //         im2col_layout_.alloc();
    //         output_not_transposed_.setDtype(inputs[0]->dtype());
    //         output_not_transposed_.reshape(inputs[0]->batch(), 1, (inputs[0]->head() / 16) * (inputs[0]->dimension() / 16), out_channel_);
    //         output_not_transposed_.alloc();
    //         outputs[0]->reshape(inputs[0]->batch(), (inputs[0]->head() / 16), out_channel_, (inputs[0]->dimension() / 16));
    //         return Op::reshape(inputs, outputs);
    //     }

    //     if (kernel_size_[0] == kernel_size_[1] && kernel_size_[0] == stride_[0] && kernel_size_[1] == stride_[1] && padding_h_ == 0 && padding_w_ == 0) {
    //         im2col_layout_.setDtype(inputs[0]->dtype());
    //         im2col_layout_.reshape(inputs[0]->batch(), 1, (inputs[0]->head() / kernel_size_[0]) * (inputs[0]->dimension() / kernel_size_[0]), kernel_size_[0] * kernel_size_[0] * in_channel_);
    //         im2col_layout_.alloc();
    //         output_not_transposed_.setDtype(inputs[0]->dtype());
    //         output_not_transposed_.reshape(inputs[0]->batch(), 1, (inputs[0]->head() / kernel_size_[0]) * (inputs[0]->dimension() / kernel_size_[0]), out_channel_);
    //         output_not_transposed_.alloc();
    //         outputs[0]->reshape(inputs[0]->batch(), (inputs[0]->head() / kernel_size_[0]), out_channel_, (inputs[0]->dimension() / kernel_size_[0]));
    //         return Op::reshape(inputs, outputs);
    //     }
    // #endif

    switch (padding_type_) {
    case SAME: {
        padding_h_ = (kernel_size_[0] - 1) / 2;
        padding_w_ = (kernel_size_[1] - 1) / 2;
        const int out_height = (inputs[0]->head() + 2 * padding_h_ - kernel_size_[0]) / stride_[0] + 1;
        const int out_width = (inputs[0]->dimension() + 2 * padding_w_ - kernel_size_[1]) / stride_[1] + 1;
        outputs[0]->reshape(inputs[0]->batch(), out_height, out_channel_, out_width);
        break;
    }
    case VALID: {
        padding_h_ = 0;
        padding_w_ = 0;
        const int out_height = (inputs[0]->head() - kernel_size_[0]) / stride_[0] + 1;
        const int out_width = (inputs[0]->dimension() - kernel_size_[1]) / stride_[1] + 1;
        outputs[0]->reshape(inputs[0]->batch(), out_height, out_channel_, out_width);
        break;
    }
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUConvolution2D::load(AbstructLoader &loader) {
    weight_.setName(name() + ".weight");
    weight_.reshape(out_channel_, kernel_size_[0], in_channel_, kernel_size_[1]);
    if (loader.getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader.getDataType(weight_.name()));
        weight_.alloc();
        loader.load(&weight_);
        // #ifndef __ARM_NEON
        kernal_ = reshape_conv2d_kernal_fp32(&weight_);
        // #endif
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
        // #ifndef __ARM_NEON
        kernal_ = reshape_conv2d_kernal_fp32(&weight_);
        // #endif
    }
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_channel_);
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

ErrorCode CPUConvolution2D::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // #ifdef __ARM_NEON
    //     if (kernel_size_[0] == 16 && kernel_size_[1] == 16 && padding_h_ == 0 && padding_w_ == 0 && stride_[0] == 16 && stride_[1] == 16) {
    //         auto start = std::chrono::high_resolution_clock::now();
    //         im2col_fp32_src_k16x16_s16_p0_to(inputs[0]->rawHostPtr(), im2col_layout_.rawHostPtr(), inputs[0]->head(), inputs[0]->dimension(), in_channel_);
    //         weight_.reshape(1, 1, out_channel_, 16 * 16 * in_channel_);
    //         mat_mul(&im2col_layout_, &weight_, &output_not_transposed_, true, &bias_, false, true, thread_count);
    //         transpose_fp32(output_not_transposed_.rawHostPtr(), outputs[0]->rawHostPtr(), (inputs[0]->head() / 16) * ((inputs[0]->dimension() / 16)), out_channel_);
    //         auto end = std::chrono::high_resolution_clock::now();
    //         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //         std::cout << duration.count() << std::endl;
    //         return Op::execute(inputs, outputs);
    //     }

    //     if (kernel_size_[0] == kernel_size_[1] && kernel_size_[0] == stride_[0] && kernel_size_[1] == stride_[1] && padding_h_ == 0 && padding_w_ == 0) {
    //         auto start = std::chrono::high_resolution_clock::now();
    //         im2col_fp32_src_knxn_sn_p0_to(inputs[0]->rawHostPtr(), im2col_layout_.rawHostPtr(), inputs[0]->head(), inputs[0]->dimension(), in_channel_, kernel_size_[0]);
    //         weight_.reshape(1, 1, out_channel_, kernel_size_[0] * kernel_size_[0] * in_channel_);
    //         mat_mul(&im2col_layout_, &weight_, &output_not_transposed_, true, &bias_, false, true, thread_count);
    //         transpose_fp32(output_not_transposed_.rawHostPtr(), outputs[0]->rawHostPtr(), (inputs[0]->head() / kernel_size_[0]) * ((inputs[0]->dimension() / kernel_size_[0])), out_channel_);
    //         auto end = std::chrono::high_resolution_clock::now();
    //         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    //         std::cout << duration.count() << std::endl;
    //         return Op::execute(inputs, outputs);
    //     }
    // #endif

    switch (padding_type_) {
    case SAME: {
        conv2d_fp32_SAME(inputs[0].get(), outputs[0].get(), kernal_, kernel_size_[0], kernel_size_[1], support_bias_, &bias_, stride_[0], stride_[1], padding_h_, padding_w_, thread_count);
        break;
    }
    case VALID: {
        conv2d_fp32_VALID(inputs[0].get(), outputs[0].get(), kernal_, kernel_size_[0], kernel_size_[1], support_bias_, &bias_, stride_[0], stride_[1], thread_count);
        break;
    }
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPUConvolution2D::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    return Op::free(inputs, outputs);
}

ErrorCode CPUConvolution2D::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}
} // namespace mllm
