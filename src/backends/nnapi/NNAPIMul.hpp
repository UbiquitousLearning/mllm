//
// Created by 30500 on 2023/10/12 0012.
//

#ifndef MLLM_NNAPIMUL_HPP
#define MLLM_NNAPIMUL_HPP

#include "NNAPICommonOp.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {
class NNAPIMul final : public NNAPICommonOp {
public:
    NNAPIMul(Backend *bn, string opName);
    virtual ~NNAPIMul() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    virtual ErrorCode load(ParamLoader &loader) override;

private:
};

class NNAPIMulCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new NNAPIMul(bn, name);
    }
};
} // namespace mllm

#endif // MLLM_NNAPIMUL_HPP
