
#ifndef MLLM_CPUSPLIT_H
#define MLLM_CPUSPLIT_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUSplit final : public Op {
public:
    CPUSplit(Backend *bn, string opName, int splitNum, Chl splitDim, int splitDimSize, int threadCount);
    virtual ~CPUSplit() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    int split_num_;
    Chl split_dim_;
    int split_dim_size_;
};

class CPUSplitCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int splitNum = (int)op_param["split_num"];
        Chl splitDim = (Chl)op_param["split_dim"];
        int splitDimSize = (int)op_param["split_dim_size"];
        return new CPUSplit(bn, name, splitNum, splitDim, splitDimSize,threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUSPLIT_H
