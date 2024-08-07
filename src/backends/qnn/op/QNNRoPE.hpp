
#ifndef MLLM_QNNROPE_H
#define MLLM_QNNROPE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNRoPE : public QNNCommonOp {
public:
    QNNRoPE(Backend *bn, string opName, int pose_type);
    virtual ~QNNRoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(AbstructLoader &loader) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:

    Tensor sin_;
    Tensor cos_;

    void sinusoidal_position_embedding(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos);
    void sinusoidal_position_embedding_hf(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos);

    int pose_type_ = 4;

    int h_cnt_ = 0;
    int pos_max_ = 16384;
    int ishape;
};

class QNNRoPECreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int pose_type = op_param["pose_type"];
        return new QNNRoPE(bn, name, pose_type);
    }
};

} // namespace mllm

#endif
