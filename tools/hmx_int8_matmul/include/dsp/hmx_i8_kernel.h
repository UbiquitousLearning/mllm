#ifndef HMX_INT8_KERNEL_H
#define HMX_INT8_KERNEL_H

#include <stdint.h>

/* Optional persistent-worker scheduler used by fused multi-head callers.
 * HMX remains on the FastRPC handler; accumulator epilogues are submitted as
 * high-priority tasks so the same workers can pack later Q/K heads whenever
 * no epilogue is ready. */
typedef void (*hmx_i8_async_task_fn)(void *context);
typedef int (*hmx_i8_async_submit_fn)(
    void *scheduler_context, hmx_i8_async_task_fn task, void *task_context);

typedef struct hmx_i8_task_scheduler {
    void *context;
    hmx_i8_async_submit_fn submit_high;
    /* Total execution lanes including the FastRPC handler. */
    int lanes;
} hmx_i8_task_scheduler;

/* Dimensions must be positive multiples of 64, 32 and 32 respectively. */
int hmx_i8_kernel(
    int32_t *output,
    const uint8_t *lhs,
    const int8_t *rhs,
    int m,
    int k,
    int n,
    uint8_t *vtcm,
    uint32_t vtcm_size);

int hmx_i8_kernel_fixed(
    int32_t *output,
    const uint8_t *lhs,
    const int8_t *rhs,
    uint8_t *vtcm,
    uint32_t vtcm_size);

int hmx_i8_kernel_i8_fixed(
    int8_t *output,
    const uint8_t *lhs,
    const int8_t *rhs,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size);

/* Q is padded [M,K]; K is padded [N,K] and transposed while being packed. */
int hmx_i8_kernel_qk_i8_fixed(
    int8_t *output,
    const uint8_t *query,
    const int8_t *key,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size);

/* Direct-path QK. Q and K are compact signed row-major tensors; signed Q is
 * converted to HMX's u8 activation domain while packing on the DSP. */
int hmx_i8_kernel_qk_i8_direct(
    int8_t *output,
    const int8_t *query,
    const int8_t *key,
    int m,
    int k,
    int n,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size);

uint32_t hmx_i8_qk_direct_vtcm_bytes(int m, int k, int n);

/* Q/K are already in HMX AH/WH tile order. K sums are signed sums over K for
 * undoing the +128 activation-domain shift. DDR-to-VTCM transfers use a
 * double-buffered DMA pipeline. */
int hmx_i8_kernel_qk_i8_packed(
    int8_t *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size);

/* CPU producer progress for the single-head, intra-head streaming path.
 * Query progress is counted in 64-row HMX tiles and K progress in 32-token
 * HMX tiles.  Value 1 remains the legacy "whole head ready" marker. */
#define HMX_I8_CPU_PACK_STREAM_FLAG 0x40000000
#define HMX_I8_CPU_PACK_QUERY_SHIFT 16
#define HMX_I8_CPU_PACK_QUERY_MASK 0x00ff0000
#define HMX_I8_CPU_PACK_KEY_MASK 0x0000ffff
/* One producer/completion word per 128-byte cache line prevents a DSP flag
 * flush from writing stale copies of progress words concurrently published
 * by the ARM producer for later heads. */
#define HMX_I8_CPU_PACK_READY_STRIDE 32
/* After a head has consumed its final input tile, the DSP flushes that
 * head's score matrix and replaces the producer progress word with this
 * sentinel.  The ARM host can therefore consume completed heads while the
 * same H-head FastRPC invocation continues executing later heads. */
#define HMX_I8_CPU_PACK_OUTPUT_READY (-2)
#define HMX_I8_CPU_PACK_OUTPUT_ERROR (-1)

/* As above, but consumes CPU-produced Q/K tiles as their progress word is
 * published.  This permits CPU pack, HMX, and the asynchronous HVX epilogue
 * to form a wavefront even when the fused group contains only one head. */
int hmx_i8_kernel_qk_i8_packed_streamed(
    int8_t *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    float output_scale,
    volatile int32_t *producer_progress,
    uint8_t *vtcm,
    uint32_t vtcm_size);

/* Packed INT8 QK using an already-running external worker scheduler for the
 * accumulator epilogue. The scheduler must remain alive until this function
 * returns; every submitted task is joined before its VTCM slot is reused. */
int hmx_i8_kernel_qk_i8_packed_scheduled(
    int8_t *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    float output_scale,
    uint8_t *vtcm,
    uint32_t vtcm_size,
    const hmx_i8_task_scheduler *scheduler);

/* Same HMX QK path, but preserves the corrected signed INT32 accumulator.
 * The output is ordinary row-major [M,N]. */
int hmx_i8_kernel_qk_i32_packed(
    int32_t *output,
    const uint8_t *packed_query,
    const int8_t *packed_key,
    const int32_t *key_sums,
    int m,
    int k,
    int n,
    uint8_t *vtcm,
    uint32_t vtcm_size);

/* Raw attention preprocessing emitted directly in HMX tile order. */
int hmx_i8_hvx_pack_query_f32_ah(
    uint8_t *packed_query,
    const float *query,
    int rows,
    int m,
    int k,
    float inverse_scale);

int hmx_i8_hvx_pack_key_f16_wh(
    int8_t *packed_key,
    int32_t *key_sums,
    const uint16_t *key,
    int n,
    int k,
    int key_begin,
    float inverse_scale);

/* DSP-side scale profiling. The input scan is performed with 128-byte HVX
 * loads; outputs are absmax / 127 and zero is represented by FLT_MIN. */
int hmx_i8_hvx_scale_f32(
    const float *input, int rows, int stride, int columns, float *scale);
int hmx_i8_hvx_scale_f16(
    const uint16_t *input, int rows, int stride, int columns, float *scale);

uint32_t hmx_i8_qk_packed_vtcm_bytes(int m, int k, int n);

#endif /* HMX_INT8_KERNEL_H */
