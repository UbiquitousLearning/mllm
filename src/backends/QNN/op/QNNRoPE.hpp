
#ifndef MLLM_QNNROPE_H
#define MLLM_QNNROPE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNRoPE : public QNNCommonOp {
public:
    QNNRoPE(Backend *bn, string opName);
    virtual ~QNNRoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNRopeCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNRoPE(bn, name);
    }
};

} // namespace mllm

#endif
