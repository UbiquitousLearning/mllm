#ifndef MLLM_CPUADD_H
#define MLLM_CPUADD_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUAdd final : public Op {
public:
    CPUAdd(Backend *bn, string opName, int threadCount);
    virtual ~CPUAdd() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUAddCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUAdd(bn, name, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUADD_H