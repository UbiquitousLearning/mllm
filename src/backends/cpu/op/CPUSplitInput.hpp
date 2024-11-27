
#ifndef MLLM_CPUSPLITINPUT_H
#define MLLM_CPUSPLITINPUT_H

#include "Op.hpp"
#include "../CPUBackend.hpp"
#include "Types.hpp"

namespace mllm {

class CPUSplitInput final : public Op {
public:
    CPUSplitInput(Backend *bn, string opName, bool isPrompt, int threadCount = 4);
    virtual ~CPUSplitInput() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    bool isPrompt_;
};

class CPUSplitInputCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new CPUSplitInput(bn, name, (bool)op_param["isPrompt"]);
    }
};

} // namespace mllm

#endif // MLLM_CPUSPLITINPUT_H
