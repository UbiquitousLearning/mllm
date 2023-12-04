
#ifndef MLLM_CPUSPLIT_H
#define MLLM_CPUSPLIT_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUSplit final : public Op {
public:
    CPUSplit(Backend *bn, string opName, int splitNum, Chl splitDim, bool multiThread);
    virtual ~CPUSplit() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool support_multi_thread_ = false;
    int split_num_;
    Chl split_dim_;
};

class CPUSplitCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int splitNum = (int)op_param["split_num"];
        Chl splitDim = (Chl)op_param["split_dim"];
        return new CPUSplit(bn, name, splitNum, splitDim, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUSPLIT_H
