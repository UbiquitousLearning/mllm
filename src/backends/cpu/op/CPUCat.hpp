
#ifndef MLLM_CPUCAT_H
#define MLLM_CPUCAT_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUCat final : public Op {
public:
    CPUCat(Backend *bn, string opName, Chl axis, int threadCount);
    virtual ~CPUCat() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    Chl axis_;
    int expd_batch_;
    int expd_batch_input_idx;
};

class CPUCatCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        const auto axis = (Chl)op_param["axis"];
        return new CPUCat(bn, name, axis, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUCAT_H
