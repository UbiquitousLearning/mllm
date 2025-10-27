#ifndef MLLM_CPUMULTIMODALROPE_H
#define MLLM_CPUMULTIMODALROPE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUMultimodalRoPE final : public Op {
public:
    CPUMultimodalRoPE(Backend *bn, string opName, float rope_theta, int max_position_embeddings, vector<int> mrope_section, int threadCount);
    
    virtual ~CPUMultimodalRoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode doExecute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs);

private:
    static vector<float> theta_; //inv_freq
    static vector<vector<float>> sin_;
    static vector<vector<float>> cos_;
    static int ishape_old;
    static int last_pos;
    vector<int> mrope_section_;
    int rope_theta_ = 10000;
    int h_cnt_ = 0;
    int pos_max_ = 16384;
    int ishape;
    int thread_count = 4;
    float partial_rotary_factor_ = 1;

    OpParam config_;

    RoPEThetaType rope_type = DEFAULT;

    void multimodal_rope_hf(shared_ptr<Tensor> input, shared_ptr<Tensor> output);
    void clearCache() override {
        h_cnt_ = 0;
    }
};

class CPUMultimodalRoPECreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        // int pose_type = op_param["pose_type"];
        // if (op_param.find("rope_theta") == op_param.end()) {
        //     return new CPUMultimodalRoPE(bn, name, pose_type, threadCount);
        // }
        // float rope_theta = op_param["rope_theta"];
        // int max_position_embeddings = op_param["max_position_embeddings"];
        // if (op_param.find("partial_rotary_factor") == op_param.end()) {
        //     return new CPUMultimodalRoPE(bn, name, pose_type, rope_theta, max_position_embeddings, threadCount);
        // }
        // float partial_rotary_factor = op_param["partial_rotary_factor"];
        // return new CPUMultimodalRoPE(bn, name, pose_type, rope_theta, partial_rotary_factor, max_position_embeddings, threadCount);

        // int pose_type = op_param["pose_type"];
        float rope_theta = op_param["rope_theta"];
        int max_position_embeddings = op_param["max_position_embeddings"];
        int length = op_param.size()-3;
        vector<int> mrope_section;
        for (int i = 0; i < length; i++) {
            mrope_section.push_back((int)op_param["mrope_section_" + std::to_string(i)]);
        }
        return new CPUMultimodalRoPE(bn, name, rope_theta, max_position_embeddings, mrope_section, threadCount);
    }
};
} // namespace mllm

#endif // MLLM_CPUMULTIMODALROPE_H