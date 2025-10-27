
#ifndef MLLM_CPUREPLACE_H
#define MLLM_CPUREPLACE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUReplace final : public Op {
public:
    CPUReplace(Backend *bn, string opName, int accumulate, int threadCount);
    ~CPUReplace() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    bool accumulate = true;
};

class CPUReplaceCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        bool accumulate = (bool)op_param["accumulate"];
        return new CPUReplace(bn, name, accumulate, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUREPLACE_H
