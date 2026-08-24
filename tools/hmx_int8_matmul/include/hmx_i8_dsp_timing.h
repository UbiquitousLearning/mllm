#ifndef HMX_I8_DSP_TIMING_H
#define HMX_I8_DSP_TIMING_H

#include <stdint.h>

enum { HMX_I8_DSP_TIMING_VERSION = 1 };

/* Durations use the always-on 19.2 MHz Hexagon qtimer. K/Q packing values are
 * sums of per-head worker time; pipeline_ticks is the corresponding overlapped
 * wall clock and must be used when constructing a critical path. */
typedef struct hmx_i8_dsp_timing {
    uint32_t struct_size;
    uint32_t version;
    uint64_t total_ticks;
    uint64_t resource_begin_ticks;
    uint64_t pipeline_ticks;
    uint64_t key_pack_work_ticks;
    uint64_t query_pack_work_ticks;
    uint64_t hmx_kernel_ticks;
    uint64_t output_flush_ticks;
    uint64_t resource_end_ticks;
} hmx_i8_dsp_timing;

static inline uint32_t hmx_i8_dsp_timing_offset(int heads) {
    const uint32_t scale_bytes = (uint32_t)heads * 3u * sizeof(float);
    return (scale_bytes + 7u) & ~7u;
}

#endif /* HMX_I8_DSP_TIMING_H */
