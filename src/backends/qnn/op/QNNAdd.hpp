#ifndef MLLM_QNNADD_H
#define MLLM_QNNADD_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNAdd : public QNNCommonOp {
public:
    QNNAdd(Backend *bn, string opName);
    virtual ~QNNAdd() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNAddCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNAdd(bn, name);
    }
};

} // namespace mllm

#endif