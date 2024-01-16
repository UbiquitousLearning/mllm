#ifndef MLLM_CPUROPE_H
#define MLLM_CPUROPE_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPURoPE final : public Op {
public:
    CPURoPE(Backend *bn, string opName, int pose_type, int threadCount);
    virtual ~CPURoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;


private:
//    Tensor freq_;
    // static Tensor sin_;
    // static Tensor cos_;
    static vector<vector<float>> sin_;
    static vector<vector<float>> cos_;
    int h_cnt_ = 0;
    int pos_max_ ;
    int pose_type_;
    int ishape;
    int thread_count = 4;
};

class CPURoPECreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int pose_type = op_param["pose_type"];
        return new CPURoPE(bn, name, pose_type, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUROPE_H