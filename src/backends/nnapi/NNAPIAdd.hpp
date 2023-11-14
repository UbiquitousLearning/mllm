#ifndef MLLM_NNAPIADD_H
#define MLLM_NNAPIADD_H

#include "NNAPICommonOp.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {

class NNAPIAdd final : public NNAPICommonOp {
public:
    NNAPIAdd(Backend *bn, string opName);
    virtual ~NNAPIAdd() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class NNAPIAddCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new NNAPIAdd(bn, name);
    }
};

} // namespace mllm

#endif // MLLM_NNAPIADD_H