
#ifndef MLLM_CPUQUICKGELU_H
#define MLLM_CPUQUICKGELU_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUQuickGELU final : public Op {
public:
    CPUQuickGELU(Backend *bn, string opName, bool multiThread);
    virtual ~CPUQuickGELU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool support_multi_thread_ = false;
};

class CPUQuickGELUCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUQuickGELU(bn, name, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUQUICKGELU_H
