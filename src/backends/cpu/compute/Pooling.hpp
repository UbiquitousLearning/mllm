//
// Created by ey on 23-12-19.
//

#ifndef POOLING_HPP
#define POOLING_HPP

#include "VecDot.hpp"
using namespace mllm;

void avgpool2d_fp32_VALID(Tensor* input, Tensor* output, int kernel_h, int kernel_w,  int stride_h, int stride_w);
void avgpool2d_fp32_SAME(Tensor* input, Tensor* output, int kernel_h, int kernel_w,  int stride_h, int stride_w, int padding_h, int padding_w);


void maxpool2d_fp32_VALID(Tensor* input, Tensor* output, int kernel_h, int kernel_w,  int stride_h, int stride_w);
void maxpool2d_fp32_SAME(Tensor* input, Tensor* output, int kernel_h, int kernel_w,  int stride_h, int stride_w, int padding_h, int padding_w);

#endif //POOLING_HPP
