#ifndef MLLM_NNAPIRMSNORM_H
#define MLLM_NNAPIRMSNORM_H

#include "NNAPIBackend.hpp"
#include "NNAPICommonOp.hpp"

namespace mllm {

class NNAPIRMSNorm final : public NNAPICommonOp {
public:
    NNAPIRMSNorm(Backend *bn, string opName, float epsilon = 1e-5);
    virtual ~NNAPIRMSNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    virtual ErrorCode load(ParamLoader &loader) override;
    Tensor &weight() {
        return weight_;
    }

private:
    bool support_multi_thread_ = false;
    float epsilon_;
    int axis_ = 1;
    Tensor weight_;
    // Tensor bias_;
};

class NNAPIRMSNormCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new NNAPIRMSNorm(bn, name);
    }
};
} // namespace mllm

#endif // MLLM_NNAPIRMSNORM_H