
#ifndef MLLM_QNNCAUSALMASK_H
#define MLLM_QNNCAUSALMASK_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNCausalMask : public QNNCommonOp {
public:
    QNNCausalMask(Backend *bn, string opName);
    virtual ~QNNCausalMask() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNCausalMaskCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNCausalMask(bn, name);
    }
};

} // namespace mllm

#endif
