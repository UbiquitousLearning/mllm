
#ifndef MLLM_QNNGELU_H
#define MLLM_QNNGELU_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNGELU : public QNNCommonOp {
public:
    QNNGELU(Backend *bn, string opName);
    virtual ~QNNGELU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    Tensor scale_;
};

class QNNGELUCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNGELU(bn, name);
    }
};

} // namespace mllm

#endif
