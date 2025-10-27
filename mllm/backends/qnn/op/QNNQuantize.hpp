
#ifndef MLLM_QNNQUANTIZE_H
#define MLLM_QNNQUANTIZE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNQuantize : public QNNCommonOp {
public:
    QNNQuantize(Backend *bn, string opName, DataType type, bool isNSHD);
    virtual ~QNNQuantize() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;

private:
    bool isNSHD_;
    Tensor scale_;

    ErrorCode setUpI8(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs);
    ErrorCode setUpI16(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs);
};

class QNNQuantizeCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNQuantize(bn, name, (DataType)op_param["dtype"], (bool)op_param["isNSHD"]);
    }
};

} // namespace mllm

#endif
