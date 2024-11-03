
#ifndef MLLM_QNNMERGEOUTPUT_H
#define MLLM_QNNMERGEOUTPUT_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNMergeOutput: public QNNCommonOp {
public:
    QNNMergeOutput(Backend *bn, string opName);
    virtual ~QNNMergeOutput() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNMergeOutputCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNMergeOutput(bn, name);
    }
};

} // namespace mllm

#endif
