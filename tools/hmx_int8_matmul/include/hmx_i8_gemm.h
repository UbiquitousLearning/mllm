#ifndef HMX_I8_GEMM_H
#define HMX_I8_GEMM_H

#include "hmx_i8_common.h"

#ifdef __cplusplus
extern "C" {
#endif

HMX_I8_API uint32_t hmx_i8_api_version(void);

/* Opens an unsigned-PD CDSP session and initializes HMX/VTCM resources. */
HMX_I8_API int hmx_i8_create(hmx_i8_context **out_context);

HMX_I8_API void hmx_i8_destroy(hmx_i8_context *context);

/* Dimensions baked into this build of libhmx_int8_matmul.so. */
HMX_I8_API int32_t hmx_i8_fixed_m(void);
HMX_I8_API int32_t hmx_i8_fixed_k(void);
HMX_I8_API int32_t hmx_i8_fixed_n(void);

/*
 * Exact signed INT8 GEMM:
 *
 *   output[m,n] = lhs[m,k] @ rhs[k,n]
 *
 * HMX natively computes u8 x i8. The implementation maps lhs to u8 with a
 * zero point of 128 and subtracts 128 * sum_k(rhs[k,n]) from every output
 * column. Inputs and output use ordinary row-major layout. Arbitrary positive
 * dimensions are accepted and padded internally to HMX's 64x32x32 tile.
 */
HMX_I8_API int hmx_i8_matmul_i32(
    hmx_i8_context *context,
    const int8_t *lhs,
    const int8_t *rhs,
    int32_t *output,
    int32_t m,
    int32_t k,
    int32_t n);

/* Same exact GEMM, with M/K/N selected at compile time. */
HMX_I8_API int hmx_i8_matmul_i32_fixed(
    hmx_i8_context *context,
    const int8_t *lhs,
    const int8_t *rhs,
    int32_t *output);

/*
 * Fixed-shape signed INT8 GEMM with DSP-side requantization:
 *
 *   accumulator = lhs @ rhs
 *   output      = clamp(round(accumulator * output_scale), -128, 127)
 *
 * output_scale is a finite, non-negative requantization multiplier. For
 * tensors whose real-value scales are lhs_scale, rhs_scale, and dst_scale,
 * pass lhs_scale * rhs_scale / dst_scale. Inputs and output are ordinary
 * row-major INT8 matrices. Signed correction, scaling, rounding, and
 * saturation are all performed on the DSP.
 */
HMX_I8_API int hmx_i8_matmul_i8_fixed(
    hmx_i8_context *context,
    const int8_t *lhs,
    const int8_t *rhs,
    int8_t *output,
    float output_scale);

/*
 * Per-tensor symmetric quantized GEMM. lhs and rhs each use one absmax scale:
 *
 *   scale = max(abs(tensor)) / 127
 *   q     = clamp(round(tensor / scale), -127, 127)
 *   output = (q_lhs @ q_rhs) * lhs_scale * rhs_scale
 *
 * A zero tensor uses scale=1 and quantizes to all zeros. Scale output pointers
 * are optional. Inputs and output are ordinary row-major float32 matrices.
 */
HMX_I8_API int hmx_i8_matmul_f32(
    hmx_i8_context *context,
    const float *lhs,
    const float *rhs,
    float *output,
    int32_t m,
    int32_t k,
    int32_t n,
    float *lhs_scale,
    float *rhs_scale);

/* Per-tensor FP32 entry point using the compile-time M/K/N. */
HMX_I8_API int hmx_i8_matmul_f32_fixed(
    hmx_i8_context *context,
    const float *lhs,
    const float *rhs,
    float *output,
    float *lhs_scale,
    float *rhs_scale);

HMX_I8_API const char *hmx_i8_status_string(int status);

#ifdef __cplusplus
}
#endif

#endif /* HMX_I8_GEMM_H */
