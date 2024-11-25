#ifndef MLLM_CPUCAUSULMASK_H
#define MLLM_CPUCAUSULMASK_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUCausalMask final : public Op {
public:
    CPUCausalMask(Backend *bn, string opName, int threadCount);
    virtual ~CPUCausalMask() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUCausalMaskCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUCausalMask(bn, name, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUCAUSULMASK_H