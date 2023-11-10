
#include "CPURoPE.hpp"
#include <cmath>

namespace mllm {

void sinusoidal_position_embedding(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
    sin.reshape({batch_size, nums_head, seq_len, output_dim});
    cos.reshape({batch_size, nums_head, seq_len, output_dim});
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
void sinusoidal_position_embedding_hf(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
    sin.reshape({batch_size, nums_head, seq_len, output_dim});
    cos.reshape({batch_size, nums_head, seq_len, output_dim});
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

CPURoPE::CPURoPE(Backend *bn, string opName, bool hf, bool multiThread) :
    Op(bn, opName) {
//    freq_.setBackend(bn);
    cos_.setBackend(bn);
    sin_.setBackend(bn);
    hf_ = hf;
}

ErrorCode CPURoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPURoPE  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    ishape = inputs[0]->shape(3);
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPURoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPURoPE()" << std::endl;
    //     auto sin_ = std::make_shared<Tensor>();
    //     auto cos_ = std::make_shared<Tensor>();
    auto &input = inputs[0];
    auto &output = outputs[0];
    for (int n = 0; n < input->shape(0); ++n) {
        for (int c = 0; c < input->shape(1); ++c) {
            for (int h = 0; h < input->shape(2); ++h) {//sequance
                #pragma omp parallel for num_threads(8)
                for (int w = 0; w < input->shape(3); ++w) {
                    if (hf_) {
                        float in_value = input->dataAt<float>(n, c, h, w);
                        float in_value_2;
                        if (w < input->shape(3) / 2) { // 偶數 0,2,4
                            in_value_2 = -input->dataAt<float>(n, c, h, w + input->shape(3) / 2);
                        } else {
                            in_value_2 = input->dataAt<float>(n, c, h, w - input->shape(3) / 2);
                        }
                        float sin_value = sin_.dataAt<float>(0, 0, h+h_cnt_, w);
                        float cos_value = cos_.dataAt<float>(0, 0, h+h_cnt_, w);
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        output->setDataAt<float>(n, c, h, w, value);
                    } else {
                        float in_value = input->dataAt<float>(n, c, h, w);
                        float in_value_2;
                        if (w % 2 == 0) { // 偶數 0,2,4
                            in_value_2 = -input->dataAt<float>(n, c, h, w + 1);
                        } else {
                            in_value_2 = input->dataAt<float>(n, c, h, w - 1);
                        }
                        float sin_value = sin_.dataAt<float>(0, 0, h+h_cnt_, w);
                        float cos_value = cos_.dataAt<float>(0, 0, h+h_cnt_, w);
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        output->setDataAt<float>(n, c, h, w, value);
                    }
                }
            }
        }
    }
    h_cnt_ += input->sequence();
    if(h_cnt_ >64){
        h_cnt_ = 0;
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPURoPE::load(ParamLoader &loader) {
//    freq_.setName("rope.freqs");
//    freq_.reshape(1, 1, 1, 64);
//    freq_.setDtype(loader.getDataType(freq_.name()));
//    freq_.alloc();
//    loader.load(&freq_);
//    freq_.printData<float>();
    if (hf_) {
        sinusoidal_position_embedding_hf(1, 1, 64, ishape, sin_, cos_);
    } else {
        sinusoidal_position_embedding(1, 1, 64, ishape, sin_, cos_);
    }
    // std::cout << name() << "  CPURoPE load" << std::endl;
    return Op::load(loader);
}
ErrorCode CPURoPE::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
//     sin_.free();
//     cos_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm
