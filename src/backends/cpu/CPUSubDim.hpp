
#ifndef MLLM_CPUSUBDIM_H
#define MLLM_CPUSUBDIM_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUSubDim final : public Op {
public:
    CPUSubDim(Backend *bn, string opName, Chl dim, vector<int> interval, bool multiThread);
    virtual ~CPUSubDim() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Chl dim_;
    int start_d_ = 999999999;
    int end_d_= 999999999;
    bool support_multi_thread_ = false;
};

class CPUSubDimCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        Chl dim = (Chl)op_param["dim"];;
        int start_i = op_param["start_i"];
        int end_i = op_param["end_i"];
        return new CPUSubDim(bn, name, dim, {start_i, end_i},false);
    }
};

} // namespace mllm

#endif // MLLM_CPUSUBDIM_H
