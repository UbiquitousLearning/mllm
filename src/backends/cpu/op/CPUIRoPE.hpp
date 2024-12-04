#ifndef MLLM_CPUIROPE_H
#define MLLM_CPUIROPE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUIRoPE final : public Op {
public:
    CPUIRoPE(Backend *bn, string opName, int pose_type, int threadCount);
    CPUIRoPE(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings, int threadCount);
    CPUIRoPE(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, int threadCount);
    virtual ~CPUIRoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode doExecute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs);

private:
    //    Tensor freq_;
    // static Tensor sin_;
    // static Tensor cos_;
    static vector<vector<int>> sin_;
    static vector<vector<int>> cos_;
    static int sin_max;
    static int cos_max;
    static int global_pose_type_;
    static int ishape_old;
    int rope_theta_ = 10000;
    int h_cnt_ = 0;
    int pos_max_ = 16384;
    int pose_type_ = 4;
    int ishape;
    int thread_count = 4;
    float partial_rotary_factor_ = 1;

    void rope_llama(shared_ptr<Tensor> input, shared_ptr<Tensor> output);
    void rope_hf(shared_ptr<Tensor> input, shared_ptr<Tensor> output);
    void rope_permission(shared_ptr<Tensor> input, shared_ptr<Tensor> output);
    void rope_mla(shared_ptr<Tensor> input, shared_ptr<Tensor> output);
    void clearCache() override {
        h_cnt_ = 0;
    }
};

class CPUIRoPECreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int pose_type = op_param["pose_type"];
        if (op_param.find("rope_theta") == op_param.end()) {
            return new CPUIRoPE(bn, name, pose_type, threadCount);
        }
        float rope_theta = op_param["rope_theta"];
        int max_position_embeddings = op_param["max_position_embeddings"];
        if (op_param.find("partial_rotary_factor") == op_param.end()) {
            return new CPUIRoPE(bn, name, pose_type, rope_theta, max_position_embeddings, threadCount);
        }
        float partial_rotary_factor = op_param["partial_rotary_factor"];
        return new CPUIRoPE(bn, name, pose_type, rope_theta, partial_rotary_factor, max_position_embeddings, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUIROPE_H