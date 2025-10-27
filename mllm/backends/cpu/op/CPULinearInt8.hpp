#ifndef MLLM_CPULINEARINT8_H
#define MLLM_CPULINEARINT8_H

#include "Op.hpp"
#include "../CPUBackend.hpp"
#include "../compute/Matmul.hpp"

namespace mllm {

class Tensor;
class CPULinearInt8 final : public Op {
public:
    CPULinearInt8(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount);
    virtual ~CPULinearInt8() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

    Tensor &weight() {
        return weight_;
    }
    Tensor &bias() {
        return bias_;
    }

private:
    int in_features_;
    int out_features_;
    bool support_bias_;
    int thread_count = 4;
    Tensor weight_;
    Tensor originWeight_;
    Tensor bias_;

    Tensor weightScale_;
    Tensor biasScale_;
    Tensor inputActivatationScale_;
    Tensor outputActivatationScale_;

    ErrorCode mat_mul_fp32_i8(Tensor *src0_, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias = nullptr, int thread_count = 4);
};

class CPULinearInt8Creator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int in_features = op_param["in_features"];
        int out_features = op_param["out_features"];
        int bias = op_param["bias"];
        return new CPULinearInt8(bn, name, in_features, out_features, (bool)bias, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPULINEAR_H