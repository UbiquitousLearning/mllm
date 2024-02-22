
#ifndef MLLM_QNNMATMULNT_H
#define MLLM_QNNMATMULNT_H

#include "QNNCommonOp.hpp"
namespace mllm {

// No k v transpose matmul.
class QNNMatmulNT : public QNNCommonOp {
public:
    QNNMatmulNT(Backend *bn, string opName, bool transpose0, bool transpose1);
    virtual ~QNNMatmulNT() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool transpose0_;
    bool transpose1_;
};

class QNNMatmulNTCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        bool transpose0 = (bool)op_param["transpose0"];
        bool transpose1 = (bool)op_param["transpose1"];
        return new QNNMatmulNT(bn, name, transpose0, transpose1);
    }
};

} // namespace mllm

#endif
