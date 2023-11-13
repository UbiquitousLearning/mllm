#ifndef MLLM_NNAPISOFTMAX_H
#define MLLM_NNAPISOFTMAX_H

#include "NNAPICommonOp.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {

class NNAPISoftMax final : public NNAPICommonOp {
public:
    NNAPISoftMax(Backend *bn, string opName, int axis);
    virtual ~NNAPISoftMax() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    virtual ErrorCode load(AbstructLoader &loader) override;

private:
    int axis_ = 0;
};

class NNAPISoftMaxCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int axis = op_param["axis"];
        return new NNAPISoftMax(bn, name, axis);
    }
};
} // namespace mllm

#endif // MLLM_NNAPISOFTMAX_H