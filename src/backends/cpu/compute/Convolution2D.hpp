//
// Created by ey on 23-12-18.
//

#ifndef CONVOLUTION2D_HPP
#define CONVOLUTION2D_HPP



#include "VecDot.hpp"
using namespace mllm;

void conv2d_fp32_VALID(Tensor* input, Tensor* output, Tensor* kernel, int stride_h, int stride_w);
void conv2d_fp32_SAME(Tensor* input, Tensor* output, Tensor* kernel, int stride_h, int stride_w, int padding_h, int padding_w);

#endif //CONVOLUTION2D_HPP
