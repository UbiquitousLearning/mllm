
#include "CPURoPE.hpp"
#include <cmath>

namespace mllm {


void sinusoidal_position_embedding(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
    sin.reshape({batch_size, nums_head, seq_len, output_dim });
    cos.reshape({batch_size, nums_head, seq_len, output_dim });
    sin.alloc();
    cos.alloc();
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < nums_head; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < output_dim; ++d) {
                    int i = (int)d/2;
                    float sin_value = std::sin(s / std::pow(10000, 2.0 * i / output_dim));
                    float cos_value = std::cos(s / std::pow(10000, 2.0 * i / output_dim));
//                    if (d % 2 == 0) {
                        sin.setDataAt<float>(n, h, s, d, sin_value);
                        cos.setDataAt<float>(n, h, s, d, cos_value);
//                    } else {
//                        sin->setDataAt<float>(n, h, s, d, cos_value);
//                        cos->setDataAt<float>(n, h, s, d, sin_value);
//                    }
                }
            }
        }
    }
}


CPURoPE::CPURoPE(Backend *bn, bool multiThread) :
    Op(bn) {
    cos_.setBackend(bn);
    sin_.setBackend(bn);
}

ErrorCode CPURoPE::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURoPE  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    sinusoidal_position_embedding(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3), sin_, cos_);
    return NO_ERROR;
}

ErrorCode CPURoPE::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURoPE  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    return NO_ERROR;
}

ErrorCode CPURoPE::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPURoPE()" << std::endl;
//    auto sin_ = std::make_shared<Tensor>();
//    auto cos_ = std::make_shared<Tensor>();
     auto & input = inputs[0];
    auto & output = outputs[0];
    for(int n = 0; n<input->shape(0); ++n){
        for(int c = 0; c<input->shape(1); ++c){
            for(int h = 0; h<input->shape(2); ++h){
                for(int w = 0; w<input->shape(3); ++w){
                    float in_value = input->dataAt<float>(n, c, h, w);
                    float sin_value = sin_.dataAt<float>(n, c, h, w);
                    float cos_value = cos_.dataAt<float>(n, c, h, w);
                    auto value = in_value * sin_value + in_value*cos_value;
                    if(w%2 ==0){
                        value = - in_value * sin_value + in_value*cos_value;
                    }
                    output->setDataAt<float>(n, c, h, w, value);
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode CPURoPE::load(ParamLoader &loader) {
    std::cout << "CPURoPE load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
