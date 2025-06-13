//
// Created by Rongjie Yi on 24-07-23.
//

#ifndef MLLM_GEMM_HPP
#define MLLM_GEMM_HPP

// #include "VecDot.hpp"
// #include "Tensor.hpp"
#include "Types.hpp"
using namespace mllm;

bool llamafile_sgemm(int64_t m, int64_t n, int64_t k, const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,
                     int ith, int nth,
                     DataType Atype, DataType Btype, DataType Ctype, void *bias = nullptr, DataType BiasType = DataType::MLLM_TYPE_F32);

bool check_llamafile_sgemm(int64_t m, int64_t n, int64_t k, DataType Atype, DataType Btype, DataType Ctype, int64_t lda, int64_t ldb, int64_t ldc);

#endif // MLLM_GEMM_HPP
