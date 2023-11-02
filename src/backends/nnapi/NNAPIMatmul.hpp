#ifndef MLLM_NNAPIMATMUL_H
#define MLLM_NNAPIMATMUL_H

#include "NNAPICommonOp.hpp"
#include "Op.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {

class Tensor;
class NNAPIMatmul final : public NNAPICommonOp {
public:
    NNAPIMatmul(Backend *bn, string opName, bool transpose0, bool transpose1);
    virtual ~NNAPIMatmul() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    virtual ErrorCode load(ParamLoader &loader) override;

private:
    bool transpose0_;
    bool transpose1_;
};

class NNAPIMatmulCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new NNAPIMatmul(bn, name, false, false);
    }
};

} // namespace mllm

#endif // MLLM_NNAPIMATMUL_H