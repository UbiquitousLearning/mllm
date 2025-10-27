#ifndef MLLM_CPUCAUSULTREEMASK_H
#define MLLM_CPUCAUSULTREEMASK_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUCausalTreeMask final : public Op {
public:
    CPUCausalTreeMask(Backend *bn, string opName, int threadCount);
    virtual ~CPUCausalTreeMask() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    // virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUCausalTreeMaskCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUCausalTreeMask(bn, name, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUCAUSULTREEMASK_H