
#ifndef MLLM_QNNRoPESimple_H
#define MLLM_QNNRoPESimple_H

#include "QNNCommonOp.hpp"
namespace mllm {
/**
 * This class should be a basic RoPE implementation for QNN backend.
 * Which only calculate value1 = in_value * cos_value - in_value_2 * sin_value and value2 = in_value * sin_value + in_value_2 * cos_value.
 * It is similar to CPUApllyRoPEFunction.
 * The sin and cos should be the inputs of this op.
 */
class QNNRoPESimple : public QNNCommonOp {
public:
    QNNRoPESimple(Backend *bn, string opName);
    virtual ~QNNRoPESimple() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
};

class QNNRoPESimpleCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNRoPESimple(bn, name);
    }
};

} // namespace mllm

#endif
