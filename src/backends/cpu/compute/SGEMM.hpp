//
// Created by Rongjie Yi on 24-07-23.
//

#ifndef MLLM_GEMM_HPP
#define MLLM_GEMM_HPP


#include "VecDot.hpp"
using namespace mllm;

bool llamafile_sgemm(int64_t, int64_t, int64_t, 
                     const void *, int64_t,
                     const void *, int64_t, 
                     void *, int64_t, 
                     DataType, DataType, DataType);

#endif // MLLM_GEMM_HPP
