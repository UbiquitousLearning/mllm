
#ifndef MLLM_QNNSUPERSILU_H
#define MLLM_QNNSUPERSILU_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNSuperSiLU : public QNNCommonOp {
public:
    QNNSuperSiLU(Backend *bn, string opName);
    virtual ~QNNSuperSiLU() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;

    Tensor a_scale_;
    Tensor b_scale_;
    Tensor o_scale_;
};


class QNNSuperSiLUCreator : public QNNBackend::Creator {

    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNSuperSiLU(bn, name);
    }
};

} // namespace mllm

#endif
