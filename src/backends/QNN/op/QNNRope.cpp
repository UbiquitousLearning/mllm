
#include "QNNRope.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNRope::QNNRope(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNRope::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());

    return NO_ERROR;
}

ErrorCode QNNRope::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    pos_max_ = 16384;
    
    ishape = inputs[0]->dimension();

    if(!sin_.allocted()) {
        if (pose_type_ == 1) {
            sinusoidal_position_embedding_hf(1, 1, pos_max_, ishape, sin_, cos_);
        } else if (pose_type_ == 2) {
            sinusoidal_position_embedding(1, 1, pos_max_, ishape, sin_, cos_);
        } else {
            sinusoidal_position_embedding_hf(1, 1, pos_max_, ishape/2, sin_, cos_);
        }
    }

    // vector<Tensor> qnn_rope_inputs;
    // qnn_rope_inputs.push_back(inputs[0])
    // qnn_rope_inputs.push_back(sin_);
    // qnn_rope_inputs.push_back(cos_);

    // return graphAddNode(name(), "RoPE", qnn_rope_inputs, outputs);

    return NO_ERROR;
}


void QNNRope::sinusoidal_position_embedding(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
    sin.reshape(batch_size, nums_head, seq_len, output_dim);
    cos.reshape(batch_size, nums_head, seq_len, output_dim);
    sin.alloc();
    cos.alloc();
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < nums_head; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < output_dim; d += 2) {
                    int i = (int)d / 2;
                    float sin_value = std::sin(s / std::pow(10000, 2.0 * i / output_dim));
                    float cos_value = std::cos(s / std::pow(10000, 2.0 * i / output_dim));
                    sin.setDataAt<float>(n, h, s, d, sin_value);
                    cos.setDataAt<float>(n, h, s, d, cos_value);
                    if (d + 1 < output_dim) {
                        sin.setDataAt<float>(n, h, s, d + 1, sin_value);
                        cos.setDataAt<float>(n, h, s, d + 1, cos_value);
                    }
                }
            }
        }
    }
}

void QNNRope::sinusoidal_position_embedding_hf(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
    sin.reshape(batch_size, nums_head, seq_len, output_dim);
    cos.reshape(batch_size, nums_head, seq_len, output_dim);
    sin.alloc();
    cos.alloc();
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < nums_head; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < output_dim; d += 2) {
                    int i = (int)d;
                    if (d >= (int)output_dim / 2) {
                        i = (int)(d - output_dim / 2);
                    }
                    float sin_value = std::sin(s / std::pow(10000, 2.0 * i / output_dim));
                    float cos_value = std::cos(s / std::pow(10000, 2.0 * i / output_dim));
                    sin.setDataAt<float>(n, h, s, d, sin_value);
                    cos.setDataAt<float>(n, h, s, d, cos_value);
                    if (d + 1 < output_dim) {
                        sin.setDataAt<float>(n, h, s, d + 1, sin_value);
                        cos.setDataAt<float>(n, h, s, d + 1, cos_value);
                    }
                }
            }
        }
    }
}

} // namespace mllm

