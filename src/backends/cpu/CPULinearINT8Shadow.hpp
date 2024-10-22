
#ifndef MLLM_CPULINEARINT8SHADOW_H
#define MLLM_CPULINEARINT8SHADOW_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {
class CPULinearINT8Shadow : public Op {
public:
    CPULinearINT8Shadow(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount);
    virtual ~CPULinearINT8Shadow() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int in_features_;
    int out_features_;
    int thread_count;
    bool support_bias_;
    Tensor weight_;
    Tensor weightScale_;
    Tensor outputScale_;
    Tensor inputScale_;

    Tensor shadowWeight_;
    Tensor shadowTransposeWeight_;

    Tensor inputClip_;
    Tensor outputClip_;

    // i16 for accuracy
    Tensor weight_f32_buffer_;
    Tensor input_f32_buffer_;

    void shadow_vec_dot_fp32_arm(float *s, float *x, int8_t *y, int n, float input_scale, float weight_scale);
};

class CPULinearINT8ShadowCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new CPULinearINT8Shadow(bn, name, in_features, out_features, (bool)bias, threadCount);
    }
};

} // namespace mllm

#endif
