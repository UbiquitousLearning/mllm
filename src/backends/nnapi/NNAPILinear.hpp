#ifndef MLLM_NNAPILINEAR_H
#define MLLM_NNAPILINEAR_H

#include "NNAPICommonOp.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {

class NNAPILinear final : public NNAPICommonOp {
public:
    NNAPILinear(Backend *bn, string opName, int in_features, int out_features, bool bias);
    virtual ~NNAPILinear() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    Tensor &weight() {
        return weight_;
    }
    Tensor &bias() {
        return bias_;
    }

private:
    int in_features_;
    int out_features_;
    bool support_bias_;
    Tensor weight_;
    Tensor bias_;
};

class NNAPILinearCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new NNAPILinear(bn, name, in_features, out_features, (bool)bias);
    }
};

} // namespace mllm

#endif // MLLM_NNAPILINEAR_H