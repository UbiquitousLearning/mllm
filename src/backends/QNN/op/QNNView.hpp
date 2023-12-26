
#ifndef MLLM_QNNVIEW_H
#define MLLM_QNNVIEW_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNView : public QNNCommonOp {
public:
    QNNView(Backend *bn, string opName);
    virtual ~QNNView() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNViewCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNView(bn, name);
    }
};

} // namespace mllm

#endif
