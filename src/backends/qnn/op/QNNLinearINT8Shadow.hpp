
#ifndef MLLM_QNNLINEARINT8SHADOW_H
#define MLLM_QNNLINEARINT8SHADOW_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNLinearINT8Shadow : public QNNCommonOp {
public:
    QNNLinearINT8Shadow(Backend *bn, string opName, int in_features, int out_features, bool bias);
    virtual ~QNNLinearINT8Shadow() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    void shadow_vec_dot_fp32_arm(float* s, float* x, int8_t* y, int n, float input_scale, float weight_scale);

private:
    int in_features_;
    int out_features_;
    bool support_bias_;
    Tensor weight_;
    Tensor weightScale_;
    Tensor outputScale_;
    Tensor inputScale_;

    Tensor shadowWeight_;
    Tensor shadowTransposeWeight_;

    Tensor inputClip_;
    Tensor outputClip_;

    Tensor output_i8_buffer_;

};

class QNNLinearINT8ShadowCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new QNNLinearINT8Shadow(bn, name, in_features, out_features, (bool)bias);
    }
};

} // namespace mllm

#endif
