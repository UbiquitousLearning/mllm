
#ifndef MLLM_QNNRMSNORM_H
#define MLLM_QNNRMSNORM_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNRMSNorm : public QNNCommonOp {
public:
    QNNRMSNorm(Backend *bn, string opName, int normSize, float epsilon = 1e-6, bool isFP32 = true);
    virtual ~QNNRMSNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;

private:
    float epsilon_;
    int axis_ = 1;
    Tensor weight_;
    int normSize_;
    bool isFP32_;
};

class QNNRMSNormCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const override {
        int normSize = (int)op_param["norm_size"];
        float epsilon = (float)op_param["epsilon"];
        bool isFP32 = (float)op_param["isFP32"];
        return new QNNRMSNorm(bn, name, normSize, epsilon, isFP32);
    }
};

} // namespace mllm

#endif
