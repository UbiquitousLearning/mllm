#ifndef MLLM_NNAPISILU_H
#define MLLM_NNAPISILU_H

#include "NNAPICommonOp.hpp"
#include "NNAPIBackend.hpp"

namespace mllm {

class NNAPISiLU final : public NNAPICommonOp {
public:
    NNAPISiLU(Backend *bn, string opName);
    virtual ~NNAPISiLU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class NNAPISiLUCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        return new NNAPISiLU(bn, name);
    }
};
} // namespace mllm

#endif // MLLM_NNAPISILU_H