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

enum { HMX_PACKED_KEY_MAP_HEADS = 16 };

static inline unsigned hmx_packed_key_source(uint64_t sources, unsigned head) {
    return (unsigned)((sources >> (head * 4u)) & 15u);
}

static inline int hmx_packed_key_map_valid(uint64_t sources, int heads) {
    if (heads <= 0 || heads > HMX_PACKED_KEY_MAP_HEADS) return 0;
    for (unsigned head = 0; head < (unsigned)heads; ++head) {
        const unsigned source = hmx_packed_key_source(sources, head);
        if (source > head || hmx_packed_key_source(sources, source) != source) {
            return 0;
        }
    }
    return 1;
}

enum hmx_i8_status {
    HMX_I8_OK = 0,
    HMX_I8_INVALID_ARGUMENT = -1,
    HMX_I8_OUT_OF_MEMORY = -2,
    HMX_I8_FASTRPC_ERROR = -3,
    HMX_I8_DSP_ERROR = -4,
    HMX_I8_SIZE_OVERFLOW = -5,
};

#endif /* HMX_I8_COMMON_H */
