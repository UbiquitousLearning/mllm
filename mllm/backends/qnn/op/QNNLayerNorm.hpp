
#ifndef MLLM_QNNLAYERNORM_H
#define MLLM_QNNLAYERNORM_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNLayerNorm : public QNNCommonOp {
public:
    QNNLayerNorm(Backend *bn, string opName, int normSize, bool bias = true, float epsilon = 1e-6);
    virtual ~QNNLayerNorm() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;

private:
    float epsilon_;
    int axis_ = 1;
    Tensor weight_;
    int normSize_;
    bool bias;
    Tensor bias_;
};

class QNNLayerNormCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const override {
        bool bias = (bool)op_param["bias"];
        int normSize = (int)op_param["norm_size"];
        float epsilon = (float)op_param["epsilon"];
        return new QNNLayerNorm(bn, name, normSize, bias, epsilon);
    }
};

} // namespace mllm

#endif
