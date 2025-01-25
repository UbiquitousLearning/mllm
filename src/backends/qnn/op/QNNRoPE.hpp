
#ifndef MLLM_QNNROPE_H
#define MLLM_QNNROPE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNRoPE : public QNNCommonOp {
public:
    QNNRoPE(Backend *bn, string opName, int pose_type);
    QNNRoPE(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings);
    QNNRoPE(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings);
    QNNRoPE(Backend *bn, string opName, OpParam &config);
    virtual ~QNNRoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
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
    float partial_rotary_factor_ = 1;

    OpParam config_;

    RoPEThetaType rope_type = DEFAULT;

    Tensor hcntTensor_;

    Tensor sinTensor_;
    Tensor cosTensor_;

    Tensor scale_;
};

class QNNRoPECreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        // from CPURoPE.cpp 2025/1/24
        auto it = op_param.find("rope_type");
        if (it != op_param.end()) {
            return new QNNRoPE(bn, name, op_param);
        }

        int pose_type = op_param["pose_type"];
        if (op_param.find("rope_theta") == op_param.end()) {
            return new QNNRoPE(bn, name, pose_type);
        }
        float rope_theta = op_param["rope_theta"];
        int max_position_embeddings = op_param["max_position_embeddings"];
        if (op_param.find("partial_rotary_factor") == op_param.end()) {
            return new QNNRoPE(bn, name, pose_type, rope_theta, max_position_embeddings);
        }
        float partial_rotary_factor = op_param["partial_rotary_factor"];
        return new QNNRoPE(bn, name, pose_type, rope_theta, partial_rotary_factor, max_position_embeddings);
    }
};

} // namespace mllm

#endif
