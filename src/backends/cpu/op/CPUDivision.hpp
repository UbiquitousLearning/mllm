
#ifndef MLLM_CPUDIVISION_H
#define MLLM_CPUDIVISION_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUDivision final : public Op {
public:
    CPUDivision(Backend *bn, string opName, int threadCount);
    virtual ~CPUDivision() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUDivisionCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUDivision(bn, name, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUDIVISION_H
