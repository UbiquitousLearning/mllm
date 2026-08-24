#ifndef HMX_I8_COMMON_H
#define HMX_I8_COMMON_H

#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define HMX_I8_API __attribute__((visibility("default")))
#else
#define HMX_I8_API
#endif

#define HMX_I8_API_VERSION 3u

typedef struct hmx_i8_context hmx_i8_context;

enum hmx_i8_status {
    HMX_I8_OK = 0,
    HMX_I8_INVALID_ARGUMENT = -1,
    HMX_I8_OUT_OF_MEMORY = -2,
    HMX_I8_FASTRPC_ERROR = -3,
    HMX_I8_DSP_ERROR = -4,
    HMX_I8_SIZE_OVERFLOW = -5,
};

#endif /* HMX_I8_COMMON_H */
