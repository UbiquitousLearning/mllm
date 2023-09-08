//
// Created by 30500 on 2020/9/29 0029.
//
#include "CPUMath.hpp"

namespace mllm {
    //C=alpha * A*B + beta*C  [M,K]*[K,N]  M K N 是转置后的维数  矩阵*矩阵
    //cblas_sgemm(CblasRowMajor, CblasNoTrans,CblasNoTrans,M,N,K,alpha,A,A的列数,B,B的列数,beta,C,C的列数)
    // template<>
    // void mllm_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    //                            const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    //                            const float alpha, const float* A, const float* B, const float beta,
    //                            float* C) {
    //     int lda = (TransA == CblasNoTrans) ? K : M;
    //     int ldb = (TransB == CblasNoTrans) ? N : K;
    //     cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                 ldb, beta, C, N);
    // }

    // template<>
    // void mllm_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    //                             const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    //                             const double alpha, const double* A, const double* B, const double beta,
    //                             double* C) {
    //     int lda = (TransA == CblasNoTrans) ? K : M;
    //     int ldb = (TransB == CblasNoTrans) ? N : K;
    //     cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                 ldb, beta, C, N);
    // }

    // //y=alpha*A*x+beta*y  矩阵*向量
    // //cblas_sgemv(CblasRowMajor, CblasNoTrans,A的行数,A的列数,alpha,A,A的列数,b,1,beta,C,1)
    // template <>
    // void mllm_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    //                            const int N, const float alpha, const float* A, const float* x,
    //                            const float beta, float* y) {
    //     cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
    // }

    // template <>
    // void mllm_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    //                             const int N, const double alpha, const double* A, const double* x,
    //                             const double beta, double* y) {
    //     cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
    // }

    // //Y=alpha * X + Y  N是X或Y的总元素长度

    // template <>
    // void mllm_cpu_axpy<float>(const int N, const float alpha, const float* X,
    //                           float* Y) {
    //     cblas_saxpy(N, alpha, X, 1, Y, 1);
    // }

    // template <>
    // void mllm_cpu_axpy<double>(const int N, const double alpha, const double* X,
    //                         double* Y) {
    //     cblas_daxpy(N, alpha, X, 1, Y, 1);
    // }

    // //Y=alpha * X + beta * Y
    // template <>
    // void mllm_cpu_axpby<float>(const int N, const float alpha, const float* X,
    //                             const float beta, float* Y) {
    //     cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
    // }

    // template <>
    // void mllm_cpu_axpby<double>(const int N, const double alpha, const double* X,
    //                              const double beta, double* Y) {
    //     cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
    // }

    // template <>
    // float mllm_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    //                                    const float* y, const int incy) {
    //     return cblas_sdot(n, x, incx, y, incy);
    // }

    // template <>
    // double mllm_cpu_strided_dot<double>(const int n, const double* x,
    //                                      const int incx, const double* y, const int incy) {
    //     return cblas_ddot(n, x, incx, y, incy);
    // }


    // template <typename Dtype>
    // Dtype mllm_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
    //     return mllm_cpu_strided_dot(n, x, 1, y, 1);
    // }

    // template
    // float mllm_cpu_dot<float>(const int n, const float* x, const float* y);

    // template
    // double mllm_cpu_dot<double>(const int n, const double* x, const double* y);

    // template <>
    // float mllm_cpu_asum<float>(const int n, const float* x) {
    //     return cblas_sasum(n, x, 1);
    // }

    // template <>
    // double mllm_cpu_asum<double>(const int n, const double* x) {
    //     return cblas_dasum(n, x, 1);
    // }

    // template <>
    // void mllm_scal<float>(const int N, const float alpha, float *X) {
    //     cblas_sscal(N, alpha, X, 1);
    // }

    // template <>
    // void mllm_scal<double>(const int N, const double alpha, double *X) {
    //     cblas_dscal(N, alpha, X, 1);
    // }


    // template <typename Dtype>
    // void mllm_set(const int N, const Dtype alpha, Dtype* Y) {
    //     if (alpha == 0) {
    //         memset(Y, 0, sizeof(Dtype) * N);
    //         return;
    //     }
    //     for (int i = 0; i < N; ++i) {
    //         Y[i] = alpha;
    //     }
    // }


    // template <>
    // void mllm_copy<float>(const int N, const float* X, float* Y) {
    //     if (X != Y) {
    //         memcpy(Y, X, sizeof(float) * N);
            
    //     }
    // }


    // template <>
    // void mllm_copy<double>(const int N, const double* X, double* Y) {
    //     if (X != Y) {
    //         memcpy(Y, X, sizeof(float) * N);
    //         }
    //     }
}



