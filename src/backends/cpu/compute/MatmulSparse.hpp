//
// Created by Rongjie Yi on 23-10-24.
//

#ifndef MLLM_MATMULSPARSE_HPP
#define MLLM_MATMULSPARSE_HPP

#include "VecDot.hpp"
using namespace mllm;

ErrorCode sparse_mat_mul_id(Tensor *x, Tensor *W, Tensor *ids, Tensor *dst, int thread_count = 4);
ErrorCode mat_mul_sparse(Tensor *x, Tensor *W, Tensor *dst, int thread_count = 4);

#endif // MLLM_MATMULSPARSE_HPP
