
#ifndef MLLM_QNNSiLUHigh_H
#define MLLM_QNNSiLUHigh_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNSiLUHigh : public QNNCommonOp {
public:
    QNNSiLUHigh(Backend *bn, string opName);
    virtual ~QNNSiLUHigh() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};


class QNNSiLUHighCreator : public QNNBackend::Creator {

    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNSiLUHigh(bn, name);
    }
};

} // namespace mllm

#endif
