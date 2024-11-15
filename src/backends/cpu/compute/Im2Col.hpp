/**
 * @file Im2Col.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-11-12
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <cstdint>

namespace mllm {

/**
 * @brief f32 Src. Kernel NxN, Stride N, Padding 0.
 *
 * C * H * W -> ((H / 16) * (W / 16)) * (N * N * C)
 *
 * !!! Dst is NOT Transposed.
 *
 * @param src
 * @param dst
 * @param H
 * @param W
 * @param C
 * @param FILTER_N
 */
void im2col_fp32_src_knxn_sn_p0_to(void *src, void *dst, int32_t H, int32_t W, int32_t C, int32_t FILTER_N);

#ifdef __ARM_NEON
/**
 * @brief  f32 Src. Kernel 16x16, Stride 16, Padding 0.
 *
 * C * H * W -> ((H / 16) * (W / 16)) * (16 * 16 * C)
 *
 * !!! Dst is NOT Transposed.
 *
 * @param src
 * @param dst
 * @param H
 * @param W
 * @param C
 */
void im2col_fp32_src_k16x16_s16_p0_to(void *src, void *dst, int32_t H, int32_t W, int32_t C);

/**
 * @brief transpose fp32 matrix
 *
 * @param src
 * @param dst
 * @param M
 * @param N
 */
void transpose_fp32(void *src, void *dst, int M, int N);
#endif //! __ARM_NEON

} // namespace mllm