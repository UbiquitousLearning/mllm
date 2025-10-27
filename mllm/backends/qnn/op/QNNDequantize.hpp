
#ifndef MLLM_QNNDEQUANTIZE_H
#define MLLM_QNNDEQUANTIZE_H

#include "QNNCommonOp.hpp"
#include "Types.hpp"
namespace mllm {
class QNNDequantize : public QNNCommonOp {
public:
    QNNDequantize(Backend *bn, string opName, bool isNSHD, bool isFP32, DataType type = MLLM_TYPE_I8);
    virtual ~QNNDequantize() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
private:
    bool isNSHD_;
    bool isFP32_;
    Tensor scale_;
    Tensor bias_;
};

class QNNDequantizeCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNDequantize(bn, name, (bool)op_param["isNSHD"], (bool)op_param["isFP32"], (DataType)op_param["inType"]);
    }
};

} // namespace mllm

#endif
