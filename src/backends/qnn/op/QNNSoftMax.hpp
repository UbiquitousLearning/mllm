
#ifndef MLLM_QNNSOFTMAX_H
#define MLLM_QNNSOFTMAX_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNSoftMax : public QNNCommonOp {
public:
    QNNSoftMax(Backend *bn, string opName, int axis);
    virtual ~QNNSoftMax() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int axis_ = 0;
};

class QNNSoftMaxCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int axis = op_param["axis"];
        return new QNNSoftMax(bn, name, axis);
    }
};

} // namespace mllm

#endif
