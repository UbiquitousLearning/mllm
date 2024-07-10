
#ifndef MLLM_QNNTRANSPOSE_H
#define MLLM_QNNTRANSPOSE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNTranspose : public QNNCommonOp {
public:
    QNNTranspose(Backend *bn, int perm0, int perm1, int perm2, int perm3, string opName);
    virtual ~QNNTranspose() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
private:
    int perm[4];
};


class QNNTransposeCreator : public QNNBackend::Creator {

    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNTranspose(bn, (int)op_param["perm0"], (int)op_param["perm1"], (int)op_param["perm2"], (int)op_param["perm3"], name);
    }
};

} // namespace mllm

#endif
