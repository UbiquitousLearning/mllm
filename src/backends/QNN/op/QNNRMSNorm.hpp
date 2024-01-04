
#ifndef MLLM_QNNRMSNORM_H
#define MLLM_QNNRMSNORM_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNRMSNorm : public QNNCommonOp {
public:
    QNNRMSNorm(Backend *bn, string opName);
    virtual ~QNNRMSNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;

private:
    float epsilon_;
    int axis_ = 1;
    Tensor weight_;
    int normSize_;
};

class QNNRMSNormCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNRMSNorm(bn, name);
    }
};

} // namespace mllm

#endif
