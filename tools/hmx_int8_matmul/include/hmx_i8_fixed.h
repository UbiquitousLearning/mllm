#ifndef HMX_I8_FIXED_H
#define HMX_I8_FIXED_H

#include "hmx_i8_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal ABI exported by libhmx_int8_fixed.so. */
HMX_I8_API uint32_t hmx_i8_api_version(void);

HMX_I8_API int hmx_i8_create(hmx_i8_context **out_context);

HMX_I8_API void hmx_i8_destroy(hmx_i8_context *context);

HMX_I8_API int32_t hmx_i8_fixed_m(void);
HMX_I8_API int32_t hmx_i8_fixed_k(void);
HMX_I8_API int32_t hmx_i8_fixed_n(void);

/*
 * output = clamp(round((lhs @ rhs) * output_scale), -128, 127)
 *
 * M/K/N are compiled into the library. output_scale is the finite,
 * non-negative accumulator-to-output multiplier. For tensor scales, pass
 * lhs_scale * rhs_scale / dst_scale.
 */
HMX_I8_API int hmx_i8_matmul_i8_fixed(
    hmx_i8_context *context,
    const int8_t *lhs,
    const int8_t *rhs,
    int8_t *output,
    float output_scale);

HMX_I8_API const char *hmx_i8_status_string(int status);

#ifdef __cplusplus
}
#endif

#endif /* HMX_I8_FIXED_H */
