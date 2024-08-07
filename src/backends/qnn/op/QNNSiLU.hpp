
#ifndef MLLM_QNNSILU_H
#define MLLM_QNNSILU_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNSiLU : public QNNCommonOp {
public:
    QNNSiLU(Backend *bn, string opName);
    virtual ~QNNSiLU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};


class QNNSiLUCreator : public QNNBackend::Creator {

    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNSiLU(bn, name);
    }
};

} // namespace mllm

#endif
