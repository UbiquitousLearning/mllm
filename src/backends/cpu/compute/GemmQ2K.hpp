#pragma once
#include "DataType.hpp"

/**
 * @brief 为 GEMM 操作，对 float 矩阵 B(KxN) 进行 Q8_K 量化和打包。
 * 将行主序的 float 矩阵 B(KxN) 转换为适合微内核高效列式访问的打包格式。
 * @param B_packed      输出，打包好的 Q8_K 矩阵。
 * @param B_float       输入，行主序的 float 矩阵 (KxN)。
 * @param K             矩阵 B 的行数。
 * @param N             矩阵 B 的列数。
 */
void quantize_and_pack_q8_k_for_gemm(
    block_q8_K *B_packed,
    const float *B_float,
    int K,
    int N);

/**
 * @brief 矩阵-向量乘法 (GEMV): y = A * x
 * @param y     输出向量 (M x 1), float
 * @param A     输入矩阵 (M x K), Q2_K 格式, 按行主序存储
 * @param x     输入向量 (K x 1), Q8_K 格式
 * @param M, K  矩阵/向量维度
 */
void gemv_q2_k_q8_k(
    float *y,
    const block_q2_K *A,
    const block_q8_K *x,
    int M,
    int K);

/**
 * @brief 矩阵-矩阵乘法 (GEMM): C = A * B
 * @param C             输出矩阵 (M x N), float, 列主序
 * @param A             输入矩阵 (M x K), Q2_K 格式, 行主序
 * @param B_packed      输入矩阵 (K x N), 已被 quantize_and_pack 处理过
 * @param M, N, K       矩阵维度
 */
void gemm_q2_k_q8_k(
    float *C,
    const block_q2_K *A,
    const block_q8_K *B_packed,
    int M,
    int N,
    int K);
