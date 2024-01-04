
#ifndef MLLM_QNNROPE_H
#define MLLM_QNNROPE_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNRope : public QNNCommonOp {
public:
    QNNRope(Backend *bn, string opName);
    virtual ~QNNRope() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:

    Tensor sin_;
    Tensor cos_;

    void sinusoidal_position_embedding(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos);
    void sinusoidal_position_embedding_hf(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos);

    int pose_type_ = 2;

    int h_cnt_ = 0;
    int pos_max_ ;
    bool hf_;
    int ishape;
    bool support_multi_thread_ = false;

};

class QNNRopeCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNRope(bn, name);
    }
};

} // namespace mllm

#endif
