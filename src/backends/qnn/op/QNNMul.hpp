
#ifndef MLLM_QNNMUL_H
#define MLLM_QNNMUL_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNMul : public QNNCommonOp {
public:
    QNNMul(Backend *bn, string opName);
    virtual ~QNNMul() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
};

class QNNMulCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNMul(bn, name);
    }
};

} // namespace mllm

#endif
