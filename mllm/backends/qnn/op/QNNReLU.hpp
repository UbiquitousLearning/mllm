
#ifndef MLLM_QNNRELU_H
#define MLLM_QNNRELU_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNReLU : public QNNCommonOp {
public:
    QNNReLU(Backend *bn, string opName);
    virtual ~QNNReLU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int axis_ = 0;
};

class QNNReLUCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNReLU(bn, name);
    }
};

} // namespace mllm

#endif
