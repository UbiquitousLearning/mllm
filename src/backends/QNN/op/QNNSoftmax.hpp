
#ifndef MLLM_QNNSOFTMAX_H
#define MLLM_QNNSOFTMAX_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNSoftmax : public QNNCommonOp {
public:
    QNNSoftmax(Backend *bn, string opName);
    virtual ~QNNSoftmax() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNSoftmaxCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNSoftmax(bn, name);
    }
};

} // namespace mllm

#endif
