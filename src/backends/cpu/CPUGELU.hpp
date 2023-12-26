
#ifndef MLLM_CPUGELU_H
#define MLLM_CPUGELU_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUGELU final : public Op {
public:
    CPUGELU(Backend *bn, string opName, bool multiThread);
    virtual ~CPUGELU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool support_multi_thread_ = false;
};

class CPUGELUCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUGELU(bn, name, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUGELU_H
