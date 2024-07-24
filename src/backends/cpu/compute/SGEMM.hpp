//
// Created by Rongjie Yi on 24-07-23.
//

#ifndef MLLM_GEMM_HPP
#define MLLM_GEMM_HPP


#include "VecDot.hpp"
using namespace mllm;

bool llamafile_sgemm(int64_t m, int64_t n, int64_t k, const void *A, int64_t lda, const void *B, int64_t ldb, void *C, int64_t ldc,                      
                    int ith, int nth,
                    DataType Atype, DataType Btype, DataType Ctype);

bool check_llamafile_sgemm(int64_t, int64_t, int64_t, 
                     DataType, DataType, DataType);


#endif // MLLM_GEMM_HPP
