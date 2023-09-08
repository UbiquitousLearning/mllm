

#ifndef MLLM_BASIC_MATH_H
#define MLLM_BASIC_MATH_H

#include <cmath>
#include <cstring>
// #include <openblas/cblas.h>
#include <limits.h>
#include <assert.h>

// #include "device.h"
// #include "common.h"

//TODO: to compile without thirdpartys

namespace mllm {
    // gemm provides a simpler interface to the gemm functions, with the
    // limitation that the data has to be contiguous in memory.
    // template <typename Dtype>
    // void mllm_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    //                     const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    //                     const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    //                     Dtype* C);
    // template <typename Dtype>
    // void mllm_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    //                     const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    //                     Dtype* y);

    template <typename Dtype>
    void mllm_cpu_axpy(const int N, const Dtype alpha, const Dtype* X,
                    Dtype* Y);

    template <typename Dtype>
    void mllm_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
                         const Dtype beta, Dtype* Y);

    template <typename Dtype>
    Dtype mllm_cpu_dot(const int n, const Dtype* x, const Dtype* y);

    template <typename Dtype>
    Dtype mllm_cpu_strided_dot(const int n, const Dtype* x, const int incx,
                                const Dtype* y, const int incy);

// Returns the sum of the absolute values of the elements of vector x
    template <typename Dtype>
    Dtype mllm_cpu_asum(const int n, const Dtype* x);

    template <typename Dtype>
    void mllm_scal(const int N, const Dtype alpha, Dtype *X);



    template <typename Dtype>
    void mllm_set(const int N, const Dtype alpha, Dtype* Y);

    // 复制数组 X到 Y
    template <typename Dtype>
    void mllm_copy(const int N, const Dtype* X, Dtype* Y);
}

#endif //MLLM_BASIC_MATH_H
