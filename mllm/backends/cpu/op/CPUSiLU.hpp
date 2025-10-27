#ifndef MLLM_CPUSILU_H
#define MLLM_CPUSILU_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUSiLU final : public Op {
public:
    CPUSiLU(Backend *bn, string opName, int threadCount);
    virtual ~CPUSiLU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUSiLUCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUSiLU(bn, name, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUSILU_H