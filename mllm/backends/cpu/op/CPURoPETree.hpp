#ifndef MLLM_CPUROPETREE_H
#define MLLM_CPUROPETREE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPURoPETree final : public Op {
public:
    CPURoPETree(Backend *bn, string opName, int pose_type, int threadCount);
    CPURoPETree(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings, int threadCount);
    CPURoPETree(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, int threadCount);
    CPURoPETree(Backend *bn, string opName, OpParam& config, int threadCount);
    virtual ~CPURoPETree() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode doExecute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs);

private:
    //    Tensor freq_;
    // static Tensor sin_;
    // static Tensor cos_;
    static vector<float> theta_;
    static vector<vector<float>> sin_;
    static vector<vector<float>> cos_;
    static int global_pose_type_;
    static int ishape_old;
    int rope_theta_ = 10000;
    int h_cnt_ = 0;
    int pos_max_ = 16384;
    int pose_type_ = 4;
    int ishape;
    int thread_count = 4;
    float partial_rotary_factor_ = 1;

    OpParam config_;

    RoPEThetaType rope_type = DEFAULT;

    void rope_llama(shared_ptr<Tensor> input, shared_ptr<Tensor> output);
    void rope_hf(shared_ptr<Tensor> input, shared_ptr<Tensor> output, shared_ptr<Tensor> tree_ancestor);
    void rope_permission(shared_ptr<Tensor> input, shared_ptr<Tensor> output);
    void rope_mla(shared_ptr<Tensor> input, shared_ptr<Tensor> output);
    void clearCache() override {
        h_cnt_ = 0;
    }
    ErrorCode updateVerifiedRoPECache(unsigned int draft_length);
};

class CPURoPETreeCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        auto it = op_param.find("rope_type");
        if (it != op_param.end()) {
            return new CPURoPETree(bn, name, op_param, threadCount);
        }

        int pose_type = op_param["pose_type"];
        if (op_param.find("rope_theta") == op_param.end()) {
            return new CPURoPETree(bn, name, pose_type, threadCount);
        }
        float rope_theta = op_param["rope_theta"];
        int max_position_embeddings = op_param["max_position_embeddings"];
        if (op_param.find("partial_rotary_factor") == op_param.end()) {
            return new CPURoPETree(bn, name, pose_type, rope_theta, max_position_embeddings, threadCount);
        }
        float partial_rotary_factor = op_param["partial_rotary_factor"];
        return new CPURoPETree(bn, name, pose_type, rope_theta, partial_rotary_factor, max_position_embeddings, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUROPETREE_H