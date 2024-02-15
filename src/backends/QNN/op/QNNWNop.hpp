#ifndef MLLM_QNNWNOP_HPP
#define MLLM_QNNWNOP_HPP

#include "QNNCommonOp.hpp"
#include "QNNBackend.hpp"

namespace mllm {
class QNNWNop final : public QNNCommonOp {
public:
    QNNWNop(Backend *bn, string opName, int sync_type);
    virtual ~QNNWNop() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    Tensor syncVar_;
    int sync_type_;
};

class QNNWNopCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNWNop(bn, name, (int)op_param["sync_type"]);
    }
};
} // namespace mllm

#endif // MLLM_CPUMUL_HPP
