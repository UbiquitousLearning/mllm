#ifndef MLLM_NNAPISCALE_H
#define MLLM_NNAPISCALE_H

#include "NNAPIBackend.hpp"
#include "NNAPICommonOp.hpp"

namespace mllm {

class NNAPIScale final : public NNAPICommonOp {
public:
    NNAPIScale(Backend *bn, string opName, float scale = 1.0, float bias = 0.0, bool bias_after_scale = true);
    virtual ~NNAPIScale() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    float scale_;
    float bias_;
    bool bias_after_scale_;
    int thread_count = 4;
};

class NNAPIScaleCreator : public NNAPIBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        // TODO: op_param :: int-->float?
        float scale = op_param["scale"];
        float bias = op_param["bias"];
        bool bias_after_scale = (bool)op_param["bias_after_scale"];
        return new NNAPIScale(bn, name, scale, bias, bias_after_scale);
    }
};

} // namespace mllm

#endif // MLLM_NNAPISCALE_H