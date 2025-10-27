
#ifndef MLLM_QNNDequantizeAdd_H
#define MLLM_QNNDequantizeAdd_H

#include "QNNCommonOp.hpp"
#include "Types.hpp"
namespace mllm {
class QNNDequantizeAdd : public QNNCommonOp {
public:
    QNNDequantizeAdd(Backend *bn, string opName, bool isNSHD, int out_features, bool isFP32, DataType type = MLLM_TYPE_I8);
    virtual ~QNNDequantizeAdd() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
private:
    bool isNSHD_;
    bool isFP32_;
    int out_features_;
    Tensor scale_;
    Tensor bias_;
};

class QNNDequantizeAddCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNDequantizeAdd(bn, name, (bool)op_param["isNSHD"], (int)op_param["out_features"], (bool)op_param["isFP32"], (DataType)op_param["inType"]);
    }
};

} // namespace mllm

#endif
