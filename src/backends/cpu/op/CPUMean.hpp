
#ifndef MLLM_CPUMEAN_H
#define MLLM_CPUMEAN_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUMean final : public Op {
public:
    CPUMean(Backend *bn, string opName, int axis, int threadCount);
    virtual ~CPUMean() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Chl axis_;
    int thread_count = 4;
};

class CPUMeanCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int axis = op_param["axis"];
        return new CPUMean(bn, name, axis, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUMEAN_H
