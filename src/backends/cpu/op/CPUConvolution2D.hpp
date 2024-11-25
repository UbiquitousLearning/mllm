
#ifndef MLLM_CPUCONVOLUTION2D_H
#define MLLM_CPUCONVOLUTION2D_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUConvolution2D final : public Op {
public:
    CPUConvolution2D(Backend *bn, string opName, int in_channel, int out_channel, vector<int> kernal_size, vector<int> stride, PaddingType padding_type, bool bias, int threadCount);
    ~CPUConvolution2D() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }

private:
    PaddingType padding_type_ = VALID;
    int stride_[2];
    int dilation_[2];
    int kernel_size_[2];
    int out_channel_;
    int in_channel_;
    int thread_count = 4;
    int padding_h_;
    int padding_w_;
    Tensor weight_;
    Tensor bias_;

#ifdef __ARM_NEON
    Tensor im2col_layout_;
    Tensor output_not_transposed_;
#endif //! __ARM_NEON

    float **kernal_;
    bool support_bias_;
};

class CPUConvolution2DCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        vector<int> kernal_size = {(int)op_param["kernal_h"], (int)op_param["kernal_w"]};
        vector<int> stride = {(int)op_param["stride_h"], (int)op_param["stride_w"]};
        int in_channel = op_param["in_channel"];
        int out_channel = op_param["out_channel"];
        PaddingType padding_type = (PaddingType)op_param["padding"];
        bool bias = (bool)op_param["bias"];
        return new CPUConvolution2D(bn, name, in_channel, out_channel, kernal_size, stride, padding_type, bias, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUCONVOLUTION2D_H
