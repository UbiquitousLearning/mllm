
#ifndef MLLM_QNNSCALE_H
#define MLLM_QNNSCALE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNScale : public QNNCommonOp {
public:
    QNNScale(Backend *bn, string opName);
    virtual ~QNNScale() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNScaleCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNScale(bn, name);
    }
};

} // namespace mllm

#endif
