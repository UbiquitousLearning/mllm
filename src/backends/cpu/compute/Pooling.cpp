//
// Created by Rongjie Yi on 23-12-19.
//

#include "Pooling.hpp"
#include "backends/cpu/third_party/ggml/VecDotFP32.hpp"
void avgpool2d_fp32_VALID(Tensor *input, Tensor *output, int kernel_h, int kernel_w, int stride_h, int stride_w, int thread_count) {
    int in_height = input->head();
    int in_width = input->dimension();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = output->sequence();
    std::vector<float> one_array(kernel_w, 1.0f);
    for (int b = 0; b < input->batch(); ++b) {
#pragma omp parallel for num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    int blk_h = out_h * stride_h;
                    int blk_w = out_w * stride_w;
                    // set value;
                    float value = 0;
                    for (int k_h = 0; k_h < kernel_h; ++k_h) {
                        float tmp_value;
                        vec_dot_fp32(kernel_w, &tmp_value, one_array.data(), input->ptrAt<float>(b, blk_h + k_h, out_ch, blk_w + 0));
                        value += tmp_value;
                    }
                    *output->ptrAt<float>(b, out_h, out_ch, out_w) = value / (kernel_h * kernel_w);
                }
            }
        }
    }
}

void avgpool2d_fp32_SAME(Tensor *input, Tensor *output, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int thread_count) {
    int padding_top = padding_h;
    int padding_left = padding_w;

    int in_height = input->head();
    int in_width = input->dimension();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = output->sequence();
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
                    if (blk_h < 0) {
                        assert(padding_top = -blk_h);
                        start_k_h += padding_top;
                    }
                    int vec_dot_n = kernel_w;
                    if (blk_w < 0) {
                        assert(padding_left = -blk_w);
                        start_k_w += padding_left;
                        vec_dot_n -= padding_left;
                    }
                    if (blk_w + kernel_w > in_width) {
                        vec_dot_n -= (blk_w + kernel_w - in_width);
                    }
                    std::vector<float> one_array(vec_dot_n, 1.0f);
                    for (int k_h = start_k_h; k_h < kernel_h & blk_h + k_h < in_height; ++k_h) {
                        float tmp_value;
                        vec_dot_fp32(vec_dot_n, &tmp_value, one_array.data(), input->ptrAt<float>(b, blk_h + k_h, out_ch, blk_w + start_k_w));
                        value += tmp_value;
                    }
                    *output->ptrAt<float>(b, out_h, out_ch, out_w) = value / (kernel_h * kernel_w);
                }
            }
        }
    }
}

void maxpool2d_fp32_VALID(Tensor *input, Tensor *output, int kernel_h, int kernel_w, int stride_h, int stride_w, int thread_count) {
    int in_height = input->head();
    int in_width = input->dimension();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = output->sequence();
    std::vector<float> one_array(kernel_w, 1.0f);
    for (int b = 0; b < input->batch(); ++b) {
#pragma omp parallel for num_threads(thread_count)
        for (int out_ch = 0; out_ch < out_channel; ++out_ch) {
            for (int out_h = 0; out_h < out_height; ++out_h) {
                for (int out_w = 0; out_w < out_width; ++out_w) {
                    int blk_h = out_h * stride_h;
                    int blk_w = out_w * stride_w;
                    // set value;
                    float value = 0;
                    for (int k_h = 0; k_h < kernel_h; ++k_h) {
                        for (int k_w = 0; k_w < kernel_w; ++k_w) {
                            value = std::max(value, *input->ptrAt<float>(b, blk_h + k_h, out_ch, blk_w + k_w));
                        }
                    }
                    *output->ptrAt<float>(b, out_h, out_ch, out_w) = value;
                }
            }
        }
    }
}
void maxpool2d_fp32_SAME(Tensor *input, Tensor *output, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int thread_count) {
    int padding_top = padding_h;
    int padding_left = padding_w;

    int in_height = input->head();
    int in_width = input->dimension();
    int out_height = output->head();
    int out_width = output->dimension();
    int out_channel = output->sequence();
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
                    float value = -999999;
                    if (blk_h < 0) {
                        assert(padding_top = -blk_h);
                        start_k_h += padding_top;
                    }
                    if (blk_w < 0) {
                        assert(padding_left = -blk_w);
                        start_k_w += padding_left;
                    }
                    for (int k_h = start_k_h; k_h < kernel_h & blk_h + k_h < in_height; ++k_h) {
                        for (int k_w = start_k_w; k_w < kernel_w & blk_w + k_w < in_width; ++k_w) {
                            value = std::max(value, *input->ptrAt<float>(b, blk_h + k_h, out_ch, blk_w + k_w));
                        }
                    }
                    *output->ptrAt<float>(b, out_h, out_ch, out_w) = value;
                }
            }
        }
    }
}