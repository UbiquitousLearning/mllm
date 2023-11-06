
#ifndef MLLM_CPUVIEW_H
#define MLLM_CPUVIEW_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUView final : public Op {
public:
    CPUView(Backend *bn, string opName, vector<int> dims, vector<int>data_dims, bool multiThread);
    virtual ~CPUView() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(ParamLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int dim0_ = -1;
    int dim1_ = -1;
    int dim2_ = -1;
    int dim3_ = -1;
    int data_dim0_;
    int data_dim1_;
    int data_dim2_;
    int data_dim3_;
    bool support_multi_thread_ = false;
};

class CPUViewCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        vector<int> dims = {op_param["dim0"], op_param["dim1"], op_param["dim2"], op_param["dim3"]};
        vector<int> data_dims = {op_param["data_dim0"], op_param["data_dim1"], op_param["data_dim2"], op_param["data_dim3"]};
        return new CPUView(bn, name, {}, {}, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUVIEW_H
