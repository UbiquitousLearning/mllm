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

} // namespace mllm

#endif