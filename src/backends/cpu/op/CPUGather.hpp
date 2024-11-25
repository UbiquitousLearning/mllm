
#ifndef MLLM_CPUGATHER_H
#define MLLM_CPUGATHER_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUGather final : public Op {
public:
    CPUGather(Backend *bn, string opName, int threadCount);
    virtual ~CPUGather() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUGatherCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUGather(bn, name, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUGATHER_H
