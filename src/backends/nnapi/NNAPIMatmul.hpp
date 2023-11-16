#ifndef MLLM_NNAPIADD_H
#define MLLM_NNAPIADD_H

#include "NNAPICommonOp.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {

class NNAPIMatmul final : public NNAPICommonOp {
public:
    NNAPIMatmul(Backend *bn, string opName);
    virtual ~NNAPIMatmul() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    // bias is always 0
    Tensor bias_;
};

class NNAPIMatmulCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new NNAPIMatmul(bn, name);
    }
};

} // namespace mllm

#endif // MLLM_NNAPIADD_H