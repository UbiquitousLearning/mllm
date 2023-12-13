
#include "CPURoPE.hpp"
#include <cmath>

namespace mllm {

void sinusoidal_position_embedding(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
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
void sinusoidal_position_embedding_hf(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
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

CPURoPE::CPURoPE(Backend *bn, string opName, int pose_type, bool multiThread) :
    Op(bn, opName) {
//    freq_.setBackend(bn);
    cos_.setBackend(bn);
    sin_.setBackend(bn);
    pose_type_ = pose_type;
}

ErrorCode CPURoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPURoPE  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    ishape = inputs[0]->dimension();
    // outputs[0]->setDtype(activationDtype());
    pos_max_ = 16384;
    if(!sin_.allocted()) {
        if (pose_type_ == 1) {
            sinusoidal_position_embedding_hf(1, 1, pos_max_, ishape, sin_, cos_);
        } else if (pose_type_ == 2) {
            sinusoidal_position_embedding(1, 1, pos_max_, ishape, sin_, cos_);
        } else {
            sinusoidal_position_embedding_hf(1, 1, pos_max_, ishape/2, sin_, cos_);
        }
    }
    return Op::reshape(inputs, outputs);
}

ErrorCode CPURoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPURoPE()" << std::endl;
    //     auto sin_ = std::make_shared<Tensor>();
    //     auto cos_ = std::make_shared<Tensor>();
    auto &input = inputs[0];
    auto &output = outputs[0];
    for (int n = 0; n < input->batch(); ++n) {
        for (int h = 0; h < input->head(); ++h) {
            for (int s = 0; s < input->sequence(); ++s) {//sequance
                #pragma omp parallel for num_threads(4)
                for (int d = 0; d < input->dimension(); ++d) {
                    if (pose_type_== 1) {
                        float in_value = input->dataAt<float>(n, h, s, d);
                        float in_value_2;
                        if (d < input->dimension() / 2) { // 偶數 0,2,4
                            in_value_2 = -input->dataAt<float>(n, h, s, d + input->dimension() / 2);
                        } else {
                            in_value_2 = input->dataAt<float>(n, h, s, d - input->dimension() / 2);
                        }
                        float sin_value = sin_.dataAt<float>(0, 0, s +h_cnt_, d);
                        float cos_value = cos_.dataAt<float>(0, 0, s +h_cnt_, d);
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                        }
                        else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                        }
                    }
                    else if (pose_type_== 2) {
                        float in_value = input->dataAt<float>(n, h, s, d);
                        float in_value_2;
                        if (d % 2 == 0) { // 偶數 0,2,4
                            in_value_2 = -input->dataAt<float>(n, h, s, d + 1);
                        } else {
                            in_value_2 = input->dataAt<float>(n, h, s, d - 1);
                        }
                        float sin_value = sin_.dataAt<float>(0, 0, s +h_cnt_, d);
                        float cos_value = cos_.dataAt<float>(0, 0, s +h_cnt_, d);
                        auto value = in_value * cos_value + in_value_2 * sin_value;
                        if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
                            output->setDataAt<float>(n, h, s, d, value);
                        }
                        else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
                            output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                        }
                    }else{
                        float in_value = input->dataAt<float>(n, h, s, d);
                        float in_value_2;
                        float sin_value = sin_.dataAt<float>(0, 0, s +h_cnt_, d);
                        float cos_value = cos_.dataAt<float>(0, 0, s +h_cnt_, d);
                        if (d < input->dimension() / 4) {
                            in_value_2 = - input->dataAt<float>(n, h, s, d + input->dimension() / 4);
                            auto value = in_value * cos_value + in_value_2 * sin_value;
                            if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
                                output->setDataAt<float>(n, h, s, d, value);
                            }
                            else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
                                output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                            }
                        } else if(d < input->dimension() / 2){
                            in_value_2 = input->dataAt<float>(n, h, s, d - input->dimension() / 4);
                            auto value = in_value * cos_value + in_value_2 * sin_value;
                            if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
                                output->setDataAt<float>(n, h, s, d, value);
                            }
                            else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
                                output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
                            }
                        }else {
                            if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
                                output->setDataAt<float>(n, h, s, d, in_value);
                            }
                            else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
                                output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(in_value));
                            }
                        }
                    }
                }
            }
        }
    }
    h_cnt_ += input->sequence();
    if(h_cnt_ >pos_max_){
        h_cnt_ = 0;
    }
    return Op::execute(inputs, outputs);
}

ErrorCode CPURoPE::load(AbstructLoader &loader) {
    //std::cout << name() << "  CPURoPE load" << std::endl;
//    freq_.setName("rope.freqs");
//    freq_.reshape(1, 1, 1, 64);
//    freq_.setDtype(loader.getDataType(freq_.name()));
//    freq_.alloc();
//    loader.load(&freq_);
//    freq_.printData<float>();
    // if (type_) {
    //     sinusoidal_position_embedding_hf(1, 1, pos_max_, ishape, sin_, cos_);
    // } else {
    //     sinusoidal_position_embedding(1, 1, pos_max_, ishape, sin_, cos_);
    // }
    // std::cout << name() << "  CPURoPE load" << std::endl;
    return Op::load(loader);
}
ErrorCode CPURoPE::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
//     sin_.free();
//     cos_.free();
    return Op::free(inputs, outputs);
}
} // namespace mllm
