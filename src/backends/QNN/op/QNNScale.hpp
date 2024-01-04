
#ifndef MLLM_QNNSCALE_H
#define MLLM_QNNSCALE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNScale : public QNNCommonOp {
public:
    QNNScale(Backend *bn, string opName, float scale=1.0, float bias=0.0, bool bias_after_scale=true);
    virtual ~QNNScale() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    float scale_;
    float bias_;
    bool bias_after_scale_;
};

class QNNScaleCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        float scale = op_param["scale"];
        float bias = op_param["bias"];
        bool bias_after_scale = (bool)op_param["bias_after_scale"];
        return new QNNScale(bn, name, scale, bias, bias_after_scale);
    }
};

} // namespace mllm

#endif
