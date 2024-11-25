
#ifndef MLLM_CPUMERGEOUTPUT_H
#define MLLM_CPUMERGEOUTPUT_H

#include "Op.hpp"
#include "../CPUBackend.hpp"
#include "Types.hpp"

namespace mllm {

class CPUMergeOutput final : public Op {
public:
    CPUMergeOutput(Backend *bn, string opName, int threadCount = 4);
    virtual ~CPUMergeOutput() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUMergeOutputCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUMergeOutput(bn, name);
    }
};

} // namespace mllm

#endif // MLLM_CPUMERGEOUTPUT_H
