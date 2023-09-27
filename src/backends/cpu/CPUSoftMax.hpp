#ifndef MLLM_CPUSOFTMAX_H
#define MLLM_CPUSOFTMAX_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUSoftMax final : public Op {
public:
    CPUSoftMax(Backend *bn, bool multiThread);
    virtual ~CPUSoftMax() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    virtual ErrorCode load(ParamLoader &loader) override;

private:
    bool support_multi_thread_ = false;
};

class CPUSoftMaxCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn) const {
        return new CPUSoftMax(bn, false);
    }
};
} // namespace mllm

#endif // MLLM_CPUSOFTMAX_H