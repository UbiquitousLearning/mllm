//
// Created by ey on 23-10-24.
//

#include "Matmul.hpp"

void vec_dot_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias_m, Tensor *bias, int hid_len, bool transpose0, bool transpose1, int batch, int head, int src0_inf, int sec1_outf){
    float value = 0;
    for (int k = 0; k < hid_len; k++) {
        // value += src0->dataAt<float>(0, h, m, k) * src1->dataAt<float>(b, h, n, k);
        if (!transpose0 && !transpose1) {
            value += src0->dataAt<float>(batch, head, src0_inf, k) * src1->dataAt<float>(batch, head, k, sec1_outf);
        } else if (transpose1) {
            value += src0->dataAt<float>(batch, head, src0_inf, k) * src1->dataAt<float>(batch, head, sec1_outf, k);
        } else {
            value += src0->dataAt<float>(batch, head, k, src0_inf) * src1->dataAt<float>(batch, head, k, sec1_outf);
        }
    }
    if (support_bias_m) {
        value += bias->dataAt<float>(0, head, 0, sec1_outf);
    }
    dst->setDataAt<float>(batch, head, src0_inf, sec1_outf, value);
}

ErrorCode mat_mul_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias_m, Tensor *bias, bool transpose0, bool transpose1) {
    // INPUT: M.K
    // W:K,N
    // OUTPUT:M.N
    //    int M = src0->sequence();
    //    int K = src0->dimension();
    //    int N = src1->dimension();
    int M = 0;
    int K = 0;
    int N = 0;
    if (!transpose0 && !transpose1) {
        M = src0->sequence();
        K = src0->dimension();
        N = src1->dimension();
    } else if (transpose1) {
        M = src0->sequence();
        K = src0->dimension();
        N = src1->sequence();
    } else {
        M = src0->dimension();
        K = src0->sequence();
        N = src1->dimension();
    }
    for (int b = 0; b < src0->batch(); b++) {
        for (int h = 0; h < src0->head(); h++) {
            for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                    vec_dot_fp32(src0, src1, dst, support_bias_m, bias, K, transpose0, transpose1, b, h, m, n);
//                    float value = 0;
//                    for (int k = 0; k < K; k++) {
//                        // value += src0->dataAt<float>(0, h, m, k) * src1->dataAt<float>(b, h, n, k);
//                        if (!transpose0 && !transpose1) {
//                            value += src0->dataAt<float>(b, h, m, k) * src1->dataAt<float>(b, h, k, n);
//                        } else if (transpose1) {
//                            value += src0->dataAt<float>(b, h, m, k) * src1->dataAt<float>(b, h, n, k);
//                        } else {
//                            value += src0->dataAt<float>(b, h, k, m) * src1->dataAt<float>(b, h, k, n);
//                        }
//                    }
//                    if (support_bias_m) {
//                        value += bias->dataAt<float>(0, h, 0, n);
//                    }
//                    dst->setDataAt<float>(b, h, m, n, value);
                }
            }
        }
    }
    return NO_ERROR;
}
ErrorCode mat_mul_fp32_q4_0(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias_m, Tensor *bias, bool transpose0, bool transpose1) {
    Tensor src1_dequantize(src1->shape());
    src1_dequantize.setDtype(src0->dtype());
    src1_dequantize.alloc();
    dequantize_row_q4_0(src1->hostPtr<block_q4_0>(), src1_dequantize.hostPtr<float>(), src1_dequantize.count());
    mat_mul_fp32(src0, &src1_dequantize, dst, support_bias_m, bias, transpose0, transpose1);
    return NO_ERROR;
}
