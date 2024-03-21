
#ifndef MLLM_CPUMATMULINT8_H
#define MLLM_CPUMATMULINT8_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUMatmulINT8 final : public Op {
public:
    CPUMatmulINT8(Backend *bn, string opName, bool transpose0, bool transpose1, int threadCount);
    virtual ~CPUMatmulINT8() override = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool transpose0_;
    bool transpose1_;
    int thread_count = 4;
    Tensor scale1_;
    Tensor scale2_;
};

class CPUMatmulINT8Creator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        bool transpose0 = (bool)op_param["transpose0"];
        bool transpose1 = (bool)op_param["transpose1"];
        return new CPUMatmulINT8(bn, name, transpose0, transpose1, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUMATMULINT8_H
