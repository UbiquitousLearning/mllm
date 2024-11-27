
#ifndef MLLM_CPUWHERE_H
#define MLLM_CPUWHERE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUWhere final : public Op {
public:
    CPUWhere(Backend *bn, string opName, float data, int axis, int threadCount);
    virtual ~CPUWhere() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    float data_;
    Chl axis_;
};

class CPUWhereCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        float data = op_param["data"];
        int axis = op_param["axis"];
        return new CPUWhere(bn, name, data, axis, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUWHERE_H
