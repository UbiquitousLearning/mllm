
#ifndef MLLM_CPUMAXPOOL_H
#define MLLM_CPUMAXPOOL_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUMaxPool2D final : public Op {
public:
    CPUMaxPool2D(Backend *bn, string opName,  vector<int> kernal_size, vector<int> stride, PaddingType padding_type, bool multiThread);
    virtual ~CPUMaxPool2D() = default;
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
    bool support_multi_thread_ = false;
};

class CPUMaxPoolCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        vector<int> kernal_size = {(int)op_param["kernal_h"],(int)op_param["kernal_w"]};
        vector<int> stride = {(int)op_param["stride_h"],(int)op_param["stride_w"]};
        PaddingType padding_type = (PaddingType)op_param["padding"];
        return new CPUMaxPool2D(bn, name, kernal_size, stride, padding_type, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUMAXPOOL_H
