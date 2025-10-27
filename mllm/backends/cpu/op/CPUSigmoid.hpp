#ifndef MLLM_CPUSIGMOID_H
#define MLLM_CPUSIGMOID_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUSigmoid final : public Op {
public:
    CPUSigmoid(Backend *bn, string opName, int threadCount);
    virtual ~CPUSigmoid() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUSigmoidCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUSigmoid(bn, name, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUSIGMOID_H