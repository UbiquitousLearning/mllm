#ifndef MLLM_CPUSOFTMAX_H
#define MLLM_CPUSOFTMAX_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUSoftMax final : public Op {
public:
    CPUSoftMax(Backend *bn, string opName, int axis, int threadCount);
    virtual ~CPUSoftMax() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int axis_ = 0;
    int thread_count = 4;
};

class CPUSoftMaxCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int axis = op_param["axis"];
        return new CPUSoftMax(bn, name, axis, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUSOFTMAX_H