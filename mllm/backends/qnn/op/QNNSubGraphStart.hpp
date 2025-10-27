
#ifndef MLLM_QNNSUBGRAPHSTART_H
#define MLLM_QNNSUBGRAPHSTART_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNSubGraphStart : public QNNCommonOp {
public:
    QNNSubGraphStart(Backend *bn, string opName);
    virtual ~QNNSubGraphStart() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNSubGraphStartCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNSubGraphStart(bn, name);
    }
};

} // namespace mllm

#endif
