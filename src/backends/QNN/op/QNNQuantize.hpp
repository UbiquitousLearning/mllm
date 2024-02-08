
#ifndef MLLM_QNNQUANTIZE_H
#define MLLM_QNNQUANTIZE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNQuantize : public QNNCommonOp {
public:
    QNNQuantize(Backend *bn, string opName);
    virtual ~QNNQuantize() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNQuantizeCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNQuantize(bn, name);
    }
};

} // namespace mllm

#endif
