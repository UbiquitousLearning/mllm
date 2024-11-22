#include "Im2Col.hpp"

#include <cstdint>
#include <cstdlib>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace mllm {

void im2col_fp32_src_knxn_sn_p0_to(void *src, void *dst, int32_t H, int32_t W, int32_t C,
                                   int32_t FILTER_N) {
    auto src_ptr = (float *)src;
    auto dst_ptr = (float *)dst;

    int32_t h_blocks = H / FILTER_N;
    int32_t w_blocks = W / FILTER_N;
    for (int32_t c = 0; c < C; ++c) {
        int D_N = 0;
        for (int32_t h = 0; h < h_blocks; ++h) {
            auto src_line_ptr = src_ptr + c * H * W + h * FILTER_N * W;
            for (int32_t w = 0; w < w_blocks; ++w) {
                auto gt_ptr = dst_ptr + c * FILTER_N * FILTER_N + D_N * FILTER_N * FILTER_N * C;
                for (int i = 0; i < FILTER_N; ++i) {
#pragma unroll
                    for (int j = 0; j < FILTER_N; ++j) {
                        *(gt_ptr + i * FILTER_N + j) = *(src_line_ptr + FILTER_N * w + i * W + j);
                    }
                }

                D_N++;
            }
        }
    }
}

#ifdef __ARM_NEON
void transpose_fp32(void *src, void *dst, int M, int N) {
    auto src_ptr = static_cast<float *>(src);
    auto dst_ptr = static_cast<float *>(dst);

    int32_t m_blocks = M / 4;
    int32_t n_blocks = N / 4;
    int32_t m_left = M % 4;
    int32_t n_left = N % 4;

    if (M > 128) {
#pragma omp parallel for num_threads(4)
        for (int32_t m = 0; m < m_blocks; ++m) {
            auto m_line_ptr = src_ptr + m * 4 * N;

            for (int32_t n = 0; n < n_blocks; ++n) {
                auto dst_line_ptr = dst_ptr + n * 4 * M;

                auto line_0 = vld1q_f32(m_line_ptr + 4 * n);
                auto line_1 = vld1q_f32(m_line_ptr + 4 * n + N);
                auto line_2 = vld1q_f32(m_line_ptr + 4 * n + 2 * N);
                auto line_3 = vld1q_f32(m_line_ptr + 4 * n + 3 * N);

                float32x4x2_t row01 = vtrnq_f32(line_0, line_1);
                float32x4x2_t row23 = vtrnq_f32(line_2, line_3);

                vst1q_f32(dst_line_ptr + 4 * m,
                          vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
                vst1q_f32(dst_line_ptr + 4 * m + M,
                          vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
                vst1q_f32(dst_line_ptr + 4 * m + 2 * M,
                          vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
                vst1q_f32(dst_line_ptr + 4 * m + 3 * M,
                          vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
            }

            if (n_left) {
                auto dst_line_ptr = dst_ptr + (n_blocks * 4 - (4 - n_left)) * M;

                auto line_0 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)));
                auto line_1 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + N);
                auto line_2 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + 2 * N);
                auto line_3 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + 3 * N);

                float32x4x2_t row01 = vtrnq_f32(line_0, line_1);
                float32x4x2_t row23 = vtrnq_f32(line_2, line_3);

                vst1q_f32(dst_line_ptr + 4 * m,
                          vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
                vst1q_f32(dst_line_ptr + 4 * m + M,
                          vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
                vst1q_f32(dst_line_ptr + 4 * m + 2 * M,
                          vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
                vst1q_f32(dst_line_ptr + 4 * m + 3 * M,
                          vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
            }
        }
    } else {
        for (int32_t m = 0; m < m_blocks; ++m) {
            auto m_line_ptr = src_ptr + m * 4 * N;

            for (int32_t n = 0; n < n_blocks; ++n) {
                auto dst_line_ptr = dst_ptr + n * 4 * M;

                auto line_0 = vld1q_f32(m_line_ptr + 4 * n);
                auto line_1 = vld1q_f32(m_line_ptr + 4 * n + N);
                auto line_2 = vld1q_f32(m_line_ptr + 4 * n + 2 * N);
                auto line_3 = vld1q_f32(m_line_ptr + 4 * n + 3 * N);

                float32x4x2_t row01 = vtrnq_f32(line_0, line_1);
                float32x4x2_t row23 = vtrnq_f32(line_2, line_3);

                vst1q_f32(dst_line_ptr + 4 * m,
                          vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
                vst1q_f32(dst_line_ptr + 4 * m + M,
                          vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
                vst1q_f32(dst_line_ptr + 4 * m + 2 * M,
                          vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
                vst1q_f32(dst_line_ptr + 4 * m + 3 * M,
                          vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
            }

            if (n_left) {
                auto dst_line_ptr = dst_ptr + (n_blocks * 4 - (4 - n_left)) * M;

                auto line_0 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)));
                auto line_1 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + N);
                auto line_2 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + 2 * N);
                auto line_3 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + 3 * N);

                float32x4x2_t row01 = vtrnq_f32(line_0, line_1);
                float32x4x2_t row23 = vtrnq_f32(line_2, line_3);

                vst1q_f32(dst_line_ptr + 4 * m,
                          vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
                vst1q_f32(dst_line_ptr + 4 * m + M,
                          vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
                vst1q_f32(dst_line_ptr + 4 * m + 2 * M,
                          vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
                vst1q_f32(dst_line_ptr + 4 * m + 3 * M,
                          vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
            }
        }
    }

    if (m_left) {
        auto m_line_ptr = src_ptr + (m_blocks * 4 - (4 - m_left)) * N;

        for (int32_t n = 0; n < n_blocks; ++n) {
            auto dst_line_ptr = dst_ptr + n * 4 * M;

            auto line_0 = vld1q_f32(m_line_ptr + 4 * n);
            auto line_1 = vld1q_f32(m_line_ptr + 4 * n + N);
            auto line_2 = vld1q_f32(m_line_ptr + 4 * n + 2 * N);
            auto line_3 = vld1q_f32(m_line_ptr + 4 * n + 3 * N);

            float32x4x2_t row01 = vtrnq_f32(line_0, line_1);
            float32x4x2_t row23 = vtrnq_f32(line_2, line_3);

            vst1q_f32(dst_line_ptr + (m_blocks * 4 - (4 - m_left)),
                      vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
            vst1q_f32(dst_line_ptr + (m_blocks * 4 - (4 - m_left)) + M,
                      vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
            vst1q_f32(dst_line_ptr + (m_blocks * 4 - (4 - m_left)) + 2 * M,
                      vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
            vst1q_f32(dst_line_ptr + (m_blocks * 4 - (4 - m_left)) + 3 * M,
                      vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
        }

        if (n_left) {
            auto dst_line_ptr = dst_ptr + (n_blocks * 4 - (4 - n_left)) * M;

            auto line_0 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)));
            auto line_1 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + N);
            auto line_2 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + 2 * N);
            auto line_3 = vld1q_f32(m_line_ptr + (n_blocks * 4 - (4 - n_left)) + 3 * N);

            float32x4x2_t row01 = vtrnq_f32(line_0, line_1);
            float32x4x2_t row23 = vtrnq_f32(line_2, line_3);

            vst1q_f32(dst_line_ptr + (m_blocks * 4 - (4 - m_left)),
                      vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0])));
            vst1q_f32(dst_line_ptr + (m_blocks * 4 - (4 - m_left)) + M,
                      vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1])));
            vst1q_f32(dst_line_ptr + (m_blocks * 4 - (4 - m_left)) + 2 * M,
                      vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0])));
            vst1q_f32(dst_line_ptr + (m_blocks * 4 - (4 - m_left)) + 3 * M,
                      vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1])));
        }
    }
}

void im2col_fp32_src_k16x16_s16_p0_to(void *src, void *dst, int32_t H, int32_t W, int32_t C) {
    int32_t h_blocks = H / 16;
    int32_t w_blocks = W / 16;
    int32_t threads = C ? C < 4 : 4;

#pragma omp parallel for num_threads(threads)
    for (int c = 0; c < C; ++c) {
        auto src_ptr = (float *)src + c * H * W;
        auto dst_ptr = (float *)dst + c * 16 * 16;

        int N = 0;
        for (int32_t h = 0; h < h_blocks; h++) {
            auto line_ptr = src_ptr + h * 16 * W;

            for (int32_t w = 0; w < w_blocks; w++) {
                auto block16x16_ptr = line_ptr + w * 16;
                auto dst_line_ptr = dst_ptr + N * 16 * 16 * C;

// process 4 x 16 four times
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float32x4x4_t line_0 = vld4q_f32(block16x16_ptr + 4 * i * W);
                    float32x4x4_t line_1 = vld4q_f32(block16x16_ptr + 4 * i * W + W);
                    float32x4x4_t line_2 = vld4q_f32(block16x16_ptr + 4 * i * W + 2 * W);
                    float32x4x4_t line_3 = vld4q_f32(block16x16_ptr + 4 * i * W + 3 * W);

                    vst4q_f32(dst_line_ptr + 64 * i, line_0);
                    vst4q_f32(dst_line_ptr + 64 * i + 16, line_1);
                    vst4q_f32(dst_line_ptr + 64 * i + 32, line_2);
                    vst4q_f32(dst_line_ptr + 64 * i + 48, line_3);
                }

                N++;
            }
        }
    }
}
#endif //! __ARM_NEON
} // namespace mllm