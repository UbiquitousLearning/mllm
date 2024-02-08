
#ifndef MLLM_QNNMATMULlINT8_H
#define MLLM_QNNMATMULlINT8_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNMatmulINT8 : public QNNCommonOp {
public:
    QNNMatmulINT8(Backend *bn, string opName, bool transpose0, bool transpose1);
    virtual ~QNNMatmulINT8() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool transpose0_;
    bool transpose1_;
};

class QNNMatmulINT8Creator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        bool transpose0 = (bool)op_param["transpose0"];
        bool transpose1 = (bool)op_param["transpose1"];
        return new QNNMatmulINT8(bn, name, transpose0, transpose1);
    }
};

} // namespace mllm

#endif
