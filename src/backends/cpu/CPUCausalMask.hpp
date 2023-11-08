#ifndef MLLM_CPUCAUSULMASK_H
#define MLLM_CPUCAUSULMASK_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUCausalMask final : public Op {
public:
    CPUCausalMask(Backend *bn, string opName, bool multiThread);
    virtual ~CPUCausalMask() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(ParamLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool support_multi_thread_ = false;
};

class CPUCausalMaskCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUCausalMask(bn, name, false);
    }
};
} // namespace mllm

#endif // MLLM_CPUCAUSULMASK_H