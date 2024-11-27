//
// Created by Rongjie Yi on 23-12-18.
//

#include "Convolution.hpp"

float **reshape_conv2d_kernal_fp32(Tensor *kernel) {
    int in_channel = kernel->sequence();
    int kernel_h = kernel->head();
    int kernel_w = kernel->dimension();
    int out_channel = kernel->batch();

    // convert the kernel to a new layout
    float **k_new = new float *[out_channel];
    for (int i = 0; i < out_channel; i++) {
        k_new[i] = new float[in_channel * kernel_h * kernel_w];
    }

    for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
        // re-layout the filter
        for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                float *kernel_p = kernel->ptrAt<float>(out_ch, k_h, in_ch, 0);
                // put all the filters in one row
                for (int i = 0; i < kernel_w; i++) {
                    k_new[out_ch][i + k_h * kernel_w + in_ch * kernel_h * kernel_w] = kernel_p[i];
                }
            }
        }
    }
    return k_new;
}

void conv2d_fp32_VALID(Tensor *input, Tensor *output, float **k_new, int kernel_h, int kernel_w, bool support_bias, Tensor *bias, int stride_h, int stride_w, int thread_count) {
    int in_channel = input->sequence();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = output->sequence();

    // convert the receptive field to a new layout
    float **i_new = new float *[out_height * out_width];
    for (int i = 0; i < out_width * out_height; i++) {
        i_new[i] = new float[in_channel * kernel_h * kernel_w];
    }

    for (int b = 0; b < input->batch(); ++b) {
        // re-layout the receptive field
        for (int out_h = 0; out_h < out_height; ++out_h) {
            for (int out_w = 0; out_w < out_width; ++out_w) {
                int blk_h = out_h * stride_h;
                int blk_w = out_w * stride_w;

                // put all the receptive field in one row
                for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
                    for (int k_h = 0; k_h < kernel_h; ++k_h) {
                        float *i_p = input->ptrAt<float>(b, blk_h + k_h, in_ch, blk_w + 0);
                        for (int i = 0; i < kernel_w; i++) {
                            i_new[out_w + out_h * out_width][i + k_h * kernel_w + in_ch * kernel_h * kernel_w] = i_p[i];
                        }
                    }
                }
            }
        }

#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    // set value;
                    float value = 0;

                    vec_dot_fp32(in_channel * kernel_h * kernel_w, &value, k_new[out_ch], i_new[out_w + out_h * out_width]);
                    if (support_bias) {
                        value += *bias->ptrAt<float>(0, 0, 0, out_ch);
                    }
                    // *output->ptrAt<float>(b, out_h, out_ch, out_w) = value;
                    output->setDataAt<float>(b, out_h, out_ch, out_w, value);
                }
            }
        }
    }
}

void conv2d_fp32_SAME(Tensor *input, Tensor *output, float **k_new, int kernel_h, int kernel_w, bool support_bias, Tensor *bias, int stride_h, int stride_w, int padding_h, int padding_w, int thread_count) {
    int padding_top = padding_h;
    int padding_left = padding_w;

    int in_height = input->head();
    int in_width = input->dimension();
    int in_channel = input->sequence();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = output->sequence();

    // 改变布局的感受野
    float **i_new = new float *[out_height * out_width];
    for (int i = 0; i < out_width * out_height; i++) {
        i_new[i] = new float[in_channel * kernel_h * kernel_w];
    }

    for (int b = 0; b < input->batch(); ++b) {
        // 将每个感受野作为一行重新布局
        for (int out_h = 0; out_h < out_height; ++out_h) {
            for (int out_w = 0; out_w < out_width; ++out_w) {
                int blk_h = out_h * stride_h - padding_top;
                int blk_w = out_w * stride_w - padding_left;

                for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
                    for (int k_h = 0; k_h < kernel_h; ++k_h) {
                        if (blk_h + k_h < 0 || blk_h + k_h >= in_height) {
                            for (int i = 0; i < kernel_w; i++) {
                                i_new[out_w + out_h * out_width][i + k_h * kernel_w + in_ch * kernel_h * kernel_w] = 0;
                            }
                        }

                        else {
                            for (int i = 0; i < kernel_w; i++) {
                                if ((blk_w + i) < 0 || (blk_w + i) >= in_width) {
                                    i_new[out_w + out_h * out_width][i + k_h * kernel_w + in_ch * kernel_h * kernel_w] = 0;

                                } else {
                                    float *i_p = input->ptrAt<float>(b, blk_h + k_h, in_ch, blk_w + i);
                                    i_new[out_w + out_h * out_width][i + k_h * kernel_w + in_ch * kernel_h * kernel_w] = (*i_p);
                                }
                            }
                        }
                    }
                }
            }
        }
#pragma omp parallel for collapse(3) num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    float value = 0;
                    vec_dot_fp32(in_channel * kernel_h * kernel_w, &value, k_new[out_ch], i_new[out_w + out_h * out_width]);

                    if (support_bias) {
                        value += *bias->ptrAt<float>(0, 0, 0, out_ch);
                    }
                    // *output->ptrAt<float>(b, out_h, out_ch, out_w) = value;
                    output->setDataAt<float>(b, out_h, out_ch, out_w, value);
                }
            }
        }
    }
}

float **reshape_conv3d_kernal_fp32(Tensor *kernel) {
    assert(kernel->ctype() == BCTHW);
    const int in_channel = kernel->channel();
    const int kernel_h = kernel->height();
    const int kernel_w = kernel->width();
    const int kernel_t = kernel->time();
    const int out_channel = kernel->batch();

    float **k_new = new float *[out_channel];
    for (int i = 0; i < out_channel; i++) {
        k_new[i] = new float[in_channel * kernel_t * kernel_h * kernel_w];
    }

    for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
        // 将过滤器重新布局
        for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
            for (int k_t = 0; k_t < kernel_t; ++k_t) {
                for (int k_h = 0; k_h < kernel_h; ++k_h) {
                    float *kernel_p = kernel->ptrAt<float>(out_ch, in_ch, k_t, k_h, 0);
                    for (int i = 0; i < kernel_w; i++) {
                        k_new[out_ch][i + k_h * kernel_w + k_t * kernel_h * kernel_w + in_ch * kernel_t * kernel_w * kernel_h] = kernel_p[i];
                    }
                }
            }
        }
    }
    return k_new;
}

void conv3d_fp32_VALID(Tensor *input, Tensor *output, float **k_new, int kernel_t, int kernel_h, int kernel_w, bool support_bias, Tensor *bias, int stride_t, int stride_h, int stride_w, int thread_count) {
    assert(input->ctype() == BCTHW);
    const int in_channel = input->channel();
    assert((output->ctype() == BCTHW || output->ctype() == BTHWC));
    const int out_height = output->height();
    const int out_width = output->width();
    const int out_time = output->time();
    const int out_channel = output->channel();

    // 改变布局的感受野
    float **i_new = new float *[out_time * out_height * out_width];
    for (int i = 0; i < out_time * out_width * out_height; i++) {
        i_new[i] = new float[in_channel * kernel_t * kernel_h * kernel_w];
    }

    for (int b = 0; b < input->batch(); ++b) {
        // 将不同的感受野重新布局
        for (int out_t = 0; out_t < out_time; ++out_t) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    int blk_t = out_t * stride_t;
                    int blk_h = out_h * stride_h;
                    int blk_w = out_w * stride_w;

                    // 将所有通道的感受野全都放在一行
                    for (int in_ch = 0; in_ch < in_channel; ++in_ch) {
                        for (int k_t = 0; k_t < kernel_t; ++k_t) {
                            for (int k_h = 0; k_h < kernel_h; ++k_h) {
                                float *i_p = input->ptrAt<float>(b, in_ch, blk_t + k_t, blk_h + k_h, blk_w + 0);
                                for (int i = 0; i < kernel_w; i++) {
                                    i_new[out_w + out_h * out_width + out_t * out_width * out_height][i + k_h * kernel_w + k_t * kernel_h * kernel_w + in_ch * kernel_t * kernel_h * kernel_w] = i_p[i];
                                }
                            }
                        }
                    }
                }
            }
        }

#pragma omp parallel for collapse(4) num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_t = 0; out_t < out_time; ++out_t) {
                for (int out_h = 0; out_h < out_height; ++out_h) {
                    for (int out_w = 0; out_w < out_width; ++out_w) {
                        // set value;
                        float value = 0;
                        vec_dot_fp32(in_channel * kernel_t * kernel_h * kernel_w, &value, k_new[out_ch], i_new[out_w + out_h * out_width + out_t * out_width * out_height]);
                        if (support_bias) {
                            value += *bias->ptrAt<float>(0, 0, 0, 0, out_ch);
                        }
                        // *output->ptrAt<float>(b, out_ch, out_t, out_h, out_w) = value;
                        output->setDataAt<float>(b, out_ch, out_t, out_h, out_w, value);
                    }
                }
            }
        }
    }
}