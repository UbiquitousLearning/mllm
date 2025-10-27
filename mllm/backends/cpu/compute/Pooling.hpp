//
// Created by Rongjie Yi on 23-12-19.
//

#ifndef POOLING_HPP
#define POOLING_HPP

#include "Tensor.hpp"
#include "Types.hpp"
using namespace mllm;

void avgpool2d_fp32_VALID(Tensor *input, Tensor *output, int kernel_h, int kernel_w, int stride_h, int stride_w, int thread_count = 4);
void avgpool2d_fp32_SAME(Tensor *input, Tensor *output, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int thread_count = 4);

void maxpool2d_fp32_VALID(Tensor *input, Tensor *output, int kernel_h, int kernel_w, int stride_h, int stride_w, int thread_count = 4);
void maxpool2d_fp32_SAME(Tensor *input, Tensor *output, int kernel_h, int kernel_w, int stride_h, int stride_w, int padding_h, int padding_w, int thread_count = 4);

#endif // POOLING_HPP
