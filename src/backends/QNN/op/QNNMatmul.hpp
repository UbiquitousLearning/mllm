
#ifndef MLLM_QNNMATMUL_H
#define MLLM_QNNMATMUL_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNMatmul : public QNNCommonOp {
public:
    QNNMatmul(Backend *bn, string opName);
    virtual ~QNNMatmul() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool transpose0_;
    bool transpose1_;
};

class QNNMatmulCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNMatmul(bn, name);
    }
};

} // namespace mllm

#endif
