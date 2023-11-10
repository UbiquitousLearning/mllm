#ifndef MLLM_NNAPI_COMMON_OP_H
#define MLLM_NNAPI_COMMON_OP_H

#include "Op.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {

class NNAPICommonOp : public Op {
public:
    NNAPICommonOp(Backend *bn);
    virtual ~NNAPICommonOp() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override = 0;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;

protected:
    NNAPIBackend *nnapiBackend_;
    std::vector<uint32_t> getTensorIdxs(const vector<shared_ptr<Tensor>> &tensors);
    template <typename T>
    inline uint32_t buildScalar(T scalar) {
        return nnapiBackend_->buildScalar(scalar);
    }
    ErrorCode buildOperation(int op, const std::vector<uint32_t> &inputs, const std::vector<uint32_t> &outputs);
    int formatAxis(int axis, const Tensor *t);
};
} // namespace mllm

#endif // MLLM_NNAPI_COMMON_OP_H