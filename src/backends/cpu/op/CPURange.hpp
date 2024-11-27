
#ifndef MLLM_CPURANGE_H
#define MLLM_CPURANGE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPURange final : public Op {
public:
    CPURange(Backend *bn, string opName, int start, int end, int threadCount);
    virtual ~CPURange() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    int start_ = 0;
    int end_;
};

class CPURangeCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int start = (int)op_param["start"];
        int end = (int)op_param["end"];
        return new CPURange(bn, name, start, end, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPURANGE_H
