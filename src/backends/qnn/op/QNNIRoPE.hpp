
#ifndef MLLM_QNNIROPE_H
#define MLLM_QNNIROPE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNIRoPE : public QNNCommonOp {
public:
    QNNIRoPE(Backend *bn, string opName, int pose_type);
    QNNIRoPE(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings);
    QNNIRoPE(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings);
    virtual ~QNNIRoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:

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
    float partial_rotary_factor_ = 1;

    Tensor hcntTensor_;

    Tensor sinTensor_;
    Tensor cosTensor_;
};

class QNNIRoPECreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int pose_type = op_param["pose_type"];
        if (op_param.find("rope_theta") == op_param.end()) {
            return new QNNIRoPE(bn, name, pose_type);
        }
        float rope_theta = op_param["rope_theta"];
        int max_position_embeddings = op_param["max_position_embeddings"];
        if (op_param.find("partial_rotary_factor") == op_param.end()) {
            return new QNNIRoPE(bn, name, pose_type, rope_theta, max_position_embeddings);
        }
        float partial_rotary_factor = op_param["partial_rotary_factor"];
        return new QNNIRoPE(bn, name, pose_type, rope_theta, partial_rotary_factor, max_position_embeddings);
    }
};

} // namespace mllm

#endif
