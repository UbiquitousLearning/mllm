
#ifndef MLLM_CPUCONVOLUTION3D_H
#define MLLM_CPUCONVOLUTION3D_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUConvolution3D final : public Op {
public:
    CPUConvolution3D(Backend *bn, string opName, int in_channel, int out_channel, vector<int> kernal_size, vector<int> stride, PaddingType padding_type, bool bias, int threadCount);
    virtual ~CPUConvolution3D() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }

private:
    PaddingType padding_type_ = VALID;
    int stride_[3];
    int dilation_[3];
    int kernel_size_[3];
    int out_channel_;
    int in_channel_;
    int thread_count = 4;
    int padding_t_;
    int padding_h_;
    int padding_w_;
    Tensor weight_;
    Tensor bias_;

    float **kernal_;
    bool support_bias_;
};

class CPUConvolution3DCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        vector<int> kernal_size = {(int)op_param["kernal_t"], (int)op_param["kernal_h"], (int)op_param["kernal_w"]};
        vector<int> stride = {(int)op_param["stride_t"], (int)op_param["stride_h"], (int)op_param["stride_w"]};
        int in_channel = op_param["in_channel"];
        int out_channel = op_param["out_channel"];
        PaddingType padding_type = (PaddingType)op_param["padding"];
        bool bias = (bool)op_param["bias"];
        return new CPUConvolution3D(bn, name, in_channel, out_channel, kernal_size, stride, padding_type, bias, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUCONVOLUTION3D_H
