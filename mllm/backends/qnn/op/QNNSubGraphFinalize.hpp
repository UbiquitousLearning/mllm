
#ifndef MLLM_QNNSUBGRAPHFINALIZE_H
#define MLLM_QNNSUBGRAPHFINALIZE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNSubGraphFinalize : public QNNCommonOp {
public:
    QNNSubGraphFinalize(Backend *bn, string opName);
    virtual ~QNNSubGraphFinalize() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNSubGraphFinalizeCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNSubGraphFinalize(bn, name);
    }
};

} // namespace mllm

#endif
