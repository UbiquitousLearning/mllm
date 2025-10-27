
#ifndef MLLM_CPUSUBDIM_H
#define MLLM_CPUSUBDIM_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUSubDim final : public Op {
public:
    CPUSubDim(Backend *bn, string opName, Chl dim, vector<int> interval, int threadCount);
    virtual ~CPUSubDim() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Chl dim_;
    int start_d_ = 999999999;
    int start_d_const_ = 999999999;
    int end_d_ = 999999999;
    int thread_count = 4;
};

class CPUSubDimCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        Chl dim = (Chl)op_param["dim"];
        ;
        int start_i = op_param["start_i"];
        int end_i = op_param["end_i"];
        return new CPUSubDim(bn, name, dim, {start_i, end_i}, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUSUBDIM_H
