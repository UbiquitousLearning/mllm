
#ifndef MLLM_QNNQUICKGELU_H
#define MLLM_QNNQUICKGELU_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNQuickGELU : public QNNCommonOp {
public:
    QNNQuickGELU(Backend *bn, string opName);
    virtual ~QNNQuickGELU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNQuickGELUCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNQuickGELU(bn, name);
    }
};

} // namespace mllm

#endif
