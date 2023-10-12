#ifndef MLLM_NNAPIADD_H
#define MLLM_NNAPIADD_H

#include "Op.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {

class NNAPIAdd final : public Op {
public:
    NNAPIAdd(Backend *bn);
    virtual ~NNAPIAdd() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;

    virtual ErrorCode load(ParamLoader &loader) override;

private:
    NNAPIBackend *nnapiBackend_;
};

class NNAPIAddCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn) const {
        return new NNAPIAdd(bn);
    }
};

} // namespace mllm

#endif // MLLM_NNAPIADD_H