//
// Created by Rongjie Yi on 23-12-18.
//

#ifndef CONVOLUTION2D_HPP
#define CONVOLUTION2D_HPP

#include "Tensor.hpp"
#include "Types.hpp"
using namespace mllm;

float **reshape_conv2d_kernal_fp32(Tensor *kernel);

void conv2d_fp32_VALID(Tensor *input, Tensor *output, float **k_new, int kernel_h, int kernel_w, bool support_bias, Tensor *bias, int stride_h, int stride_w, int thread_count = 4);
void conv2d_fp32_SAME(Tensor *input, Tensor *output, float **k_new, int kernel_h, int kernel_w, bool support_bias, Tensor *bias, int stride_h, int stride_w, int padding_h, int padding_w, int thread_count = 4);

float **reshape_conv3d_kernal_fp32(Tensor *kernel);

void conv3d_fp32_VALID(Tensor *input, Tensor *output, float **k_new, int kernel_t, int kernel_h, int kernel_w, bool support_bias, Tensor *bias, int stride_t, int stride_h, int stride_w, int thread_count = 4);

#endif // CONVOLUTION2D_HPP
