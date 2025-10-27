
#ifndef MLLM_CPUAVGPOOL_H
#define MLLM_CPUAVGPOOL_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUAvgPool2D final : public Op {
public:
    CPUAvgPool2D(Backend *bn, string opName, vector<int> kernal_size, vector<int> stride, PaddingType padding_type, int threadCount);
    virtual ~CPUAvgPool2D() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    PaddingType padding_type_ = VALID;
    int stride_[2];
    int dilation_[2];
    int kernel_size_[2];
    int padding_h_;
    int padding_w_;
    int thread_count = 4;
};

class CPUAvgPoolCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        vector<int> kernal_size = {(int)op_param["kernal_h"], (int)op_param["kernal_w"]};
        vector<int> stride = {(int)op_param["stride_h"], (int)op_param["stride_w"]};
        PaddingType padding_type = (PaddingType)op_param["padding"];
        return new CPUAvgPool2D(bn, name, kernal_size, stride, padding_type, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUAVGPOOL_H
