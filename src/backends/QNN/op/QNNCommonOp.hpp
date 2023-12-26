#ifndef MLLM_QNN_COMMON_OP_H
#define MLLM_QNN_COMMON_OP_H

#include "Op.hpp"
#include "QNNBackend.hpp"
#include "QnnTypes.h"
#include "Types.hpp"

namespace mllm {

class QNNCommonOp : public Op {
public:
    QNNCommonOp(Backend *bn, string opName);
    virtual ~QNNCommonOp() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override = 0;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;

protected:
    QNNBackend *qnnBackend_;
    ErrorCode graphAddNode(string name, string nodeType, vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, vector<Qnn_Param_t> params = {}, string packageName = "qti.aisw");
};
} // namespace mllm

#endif // MLLM_QNN_COMMON_OP_H