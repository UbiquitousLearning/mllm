//
// Created by ey on 23-12-18.
//

#include "Convolution.hpp"

void conv2d_fp32_VALID(Tensor* input, Tensor* output, Tensor* kernel, bool support_bias, Tensor* bias, int stride_h, int stride_w, int thread_count) {
    int in_height = input->head();
    int in_width = input->dimension();
    int in_channel = input->sequence();
    int kernel_h = kernel->head();
    int kernel_w = kernel->dimension();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = kernel->batch();
    for (int b = 0; b < input->batch(); ++b) {
#pragma omp parallel for num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    int blk_h = out_h * stride_h;
                    int blk_w = out_w * stride_w;
                    // set value;
                    float value = 0;
                    for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
                            float* kernel_p = kernel->ptrAt<float>(out_ch, k_h, in_ch, 0);
                            float tmp_value;
                            vec_dot_fp32(kernel_w, &tmp_value, kernel_p, input->ptrAt<float>(b, blk_h+k_h, in_ch, blk_w+0));
                            value += tmp_value;
                        }
                    }
                    if (support_bias) {
                        value += *bias->ptrAt<float>(0, 0, 0, out_ch);
                    }
                    *output->ptrAt<float>(b, out_h, out_ch, out_w) = value;
                }
            }
        }
    }
}

void conv2d_fp32_q4_K_VALID(Tensor* input_, Tensor* output, Tensor* kernel, bool support_bias, Tensor* bias, int stride_h, int stride_w, int thread_count) {
    assert(kernel->dtype() == MLLM_TYPE_Q4_K);
    assert(input_->dtype() == MLLM_TYPE_F32);

    Tensor src0_q8(input_->shape());
    src0_q8.setBackend(input_->backend());
    src0_q8.setDtype(MLLM_TYPE_Q8_K);
    src0_q8.alloc();
    if (input_->dimension() % QK_K == 0) {
        for (int b = 0; b < input_->batch(); b++) {
            for (int h = 0; h < input_->head(); h++) {
#pragma omp parallel for num_threads(thread_count)
                for (int s = 0; s < input_->sequence(); s++) {
                    quantize_row_q8_K(input_->hostPtr<float>() + input_->offset(b, h, s, 0),
                                      src0_q8.hostPtr<block_q8_K>() + src0_q8.offset(b, h, s, 0) / QK_K,
                                      input_->dimension());
                }
            }
        }
    } else {
        std::cout << "[ERROR]: " << input_->dimension() << "%" << QK_K << "!=0" << std::endl;
        assert(input_->dimension() % QK_K == 0);
    }
    auto *input= &src0_q8;
    int in_height = input->head();
    int in_width = input->dimension();
    int in_channel = input->sequence();
    int kernel_h = kernel->head();
    int kernel_w = kernel->dimension();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = kernel->batch();
    for (int b = 0; b < input->batch(); ++b) {
#pragma omp parallel for num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    int blk_h = out_h * stride_h;
                    int blk_w = out_w * stride_w;
                    // set value;
                    float value = 0;
                    for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
                        for (int k_h = 0; k_h < kernel_h; ++k_h) {
                            // float* kernel_p = kernel->ptrAt<float>(out_ch, k_h, in_ch, 0);
                            float tmp_value;
                            // vec_dot_fp32(kernel_w, &tmp_value, kernel_p,
                            //             input->ptrAt<float>(b, blk_h+k_h, in_ch, blk_w+0));
                            vec_dot_q4_K_q8_K(kernel_w, &tmp_value,
                                              kernel->hostPtr<block_q4_K>() + kernel->offset(out_ch, k_h, in_ch, 0) / QK_K,
                                              input->hostPtr<block_q8_K>() + input->offset(b, blk_h+k_h, in_ch, blk_w+0) / QK_K);
                            value += tmp_value;
                        }
                    }
                    if (support_bias) {
                        value += *bias->ptrAt<float>(0, 0, 0, out_ch);
                    }
                    *output->ptrAt<float>(b, out_h, out_ch, out_w) = value;
                }
            }
        }
    }
}

void conv2d_fp32_SAME(Tensor* input, Tensor* output, Tensor* kernel, bool support_bias, Tensor* bias, int stride_h, int stride_w, int padding_h, int padding_w, int thread_count) {
    int padding_top = padding_h ;
    int padding_left = padding_w ;
    
    int in_height = input->head();
    int in_width = input->dimension();
    int in_channel = input->sequence();
    int kernel_h = kernel->head();
    int kernel_w = kernel->dimension();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = kernel->batch();
    for (int b = 0; b < input->batch(); ++b) {
#pragma omp parallel for num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    int blk_h = out_h * stride_h - padding_top;
                    int blk_w = out_w * stride_w - padding_left;
                    // set value;
                    int start_k_h = 0;
                    int start_k_w = 0;
                    float value = 0;
                    if(blk_h<0) {
                        assert(padding_top = -blk_h);
                        start_k_h += padding_top;
                    }
                    int vec_dot_n = kernel_w;
                    if (blk_w<0) {
                        assert(padding_left = -blk_w);
                        start_k_w += padding_left;
                        vec_dot_n -= padding_left;
                    }
                    if(blk_w+ kernel_w > in_width) {
                        vec_dot_n -= (blk_w+ kernel_w - in_width);
                    }
                    for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
                        for (int k_h = start_k_h; k_h < kernel_h & blk_h+k_h<in_height ; ++k_h) {
                            float* kernel_p = kernel->ptrAt<float>(out_ch, k_h, in_ch, 0+start_k_w);
                            float tmp_value;
                            vec_dot_fp32(vec_dot_n, &tmp_value, kernel_p, input->ptrAt<float>(b, blk_h+k_h, in_ch, blk_w+start_k_w));
                            value += tmp_value;
                        }
                    }
                    if (support_bias) {
                        value += *bias->ptrAt<float>(0, 0, 0, out_ch);
                    }
                    *output->ptrAt<float>(b, out_h, out_ch, out_w) = value;
                }
            }
        }
    }
}

void conv3d_fp32_VALID(Tensor* input, Tensor* output, Tensor* kernel, bool support_bias, Tensor* bias, int stride_t, int stride_h, int stride_w, int thread_count) {
    assert(input->ctype() == BCTHW);
    const int in_height = input->head();
    const int in_width = input->width();
    const int in_time = input->time();
    const int in_channel = input->channel();
    assert(in_channel == kernel->channel());
    assert(kernel->ctype() == BCTHW);
    const int kernel_h = kernel->height();
    const int kernel_w = kernel->width();
    const int kernel_t = kernel->time();
    assert((output->ctype() == BCTHW || output->ctype() == BTHWC));
    const int out_height = output->height();
    const int out_width = output->width();
    const int out_time = output->time();
    const int out_channel = output->channel();
    assert(out_channel == kernel->batch());
    for (int b = 0; b < input->batch(); ++b) {
#pragma omp parallel for num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_t = 0; out_t < out_time; ++out_t) {
                for (int out_h = 0; out_h < out_height; ++out_h) {
                    for (int out_w = 0; out_w < out_width; ++out_w) {
                        int blk_t = out_t * stride_t;
                        int blk_h = out_h * stride_h;
                        int blk_w = out_w * stride_w;
                        // set value;
                        float value = 0;
                        for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
                            for (int k_t = 0; k_t < kernel_t; ++k_t) {
                                for (int k_h = 0; k_h < kernel_h; ++k_h) {
                                    float tmp_value;
                                    vec_dot_fp32(kernel_w, &tmp_value,
                                        kernel->ptrAt<float>(out_ch,  in_ch, k_t, k_h,0),
                                        input->ptrAt<float>(b, in_ch, blk_t+k_t, blk_h + k_h, blk_w + 0));
                                    value += tmp_value;
                                    // for (int k_w = 0; k_w < kernel_w; ++k_w) {
                                    //     value += kernel->dataAt<float>(out_ch,  in_ch, k_t, k_h,k_w) *
                                    //         input->dataAt<float>(b, in_ch, blk_t+k_t, blk_h + k_h, blk_w + k_w);
                                    // }
                                }
                            }
                        }
                        if (support_bias) {
                            value += *bias->ptrAt<float>(0, 0, 0, 0, out_ch);
                        }
                        *output->ptrAt<float>(b, out_ch, out_t, out_h, out_w) = value;
                    }
                }
            }
        }
    }
}
