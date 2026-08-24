#ifndef HMX_QK_I8_OPERATOR_H
#define HMX_QK_I8_OPERATOR_H

#include "hmx_i8_common.h"
#include "hmx_i8_dsp_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HMX_QK_I8_OPERATOR_API_VERSION 5u

enum {
    HMX_QK_I8_FLAG_PERSISTENT_RPCMEM = 1u << 0,
    HMX_QK_I8_FLAG_DIRECT_IO = 1u << 1,
    HMX_QK_I8_FLAG_N_IS_CAPACITY = 1u << 2,
    HMX_QK_I8_FLAG_HEADS_IS_CAPACITY = 1u << 3,
    HMX_QK_I8_FLAG_DSP_LAYOUT = 1u << 4,
    HMX_QK_I8_FLAG_PREPACKED_QK = 1u << 5,
    HMX_QK_I8_FLAG_KEY_SUMS = 1u << 6,
    HMX_QK_I8_FLAG_EXECUTION_SCOPE = 1u << 7,
    HMX_QK_I8_FLAG_DMA_PIPELINE = 1u << 8,
    HMX_QK_I8_FLAG_RAW_QK_HVX = 1u << 9,
    HMX_QK_I8_FLAG_INCREMENTAL_K = 1u << 10,
    HMX_QK_I8_FLAG_HVX_SCALE_PROFILE = 1u << 11,
    HMX_QK_I8_FLAG_PERSISTENT_DSP_ARENA = 1u << 12,
    HMX_QK_I8_FLAG_FUSED_RAW_QK = 1u << 13,
    HMX_QK_I8_FLAG_PER_HEAD_BUCKET_SCALES = 1u << 14,
    HMX_QK_I8_FLAG_CPU_PACKED_PIPELINE = 1u << 15,
    HMX_QK_I8_FLAG_LONG_RPC_HEAD_READY = 1u << 16,
};

/* Metadata embedded in every fixed shape/scale operator shared library. */
typedef struct hmx_qk_i8_operator_info {
    uint32_t struct_size;
    int32_t heads;
    int32_t m;
    int32_t k;
    int32_t n;
    float q_scale;
    float k_scale;
    float output_scale;
    float requant_scale;
    uint32_t flags;
} hmx_qk_i8_operator_info;

HMX_I8_API uint32_t hmx_qk_i8_operator_api_version(void);

HMX_I8_API int hmx_qk_i8_operator_get_info(
    hmx_qk_i8_operator_info *info);

HMX_I8_API int hmx_i8_create(hmx_i8_context **out_context);

HMX_I8_API void hmx_i8_destroy(hmx_i8_context *context);

/*
 * Fixed Q[H,M,K] @ transpose(K[H,N,K]) -> score[H,M,N]. Batch is fixed to 1,
 * and all tensors use row-major BHSD order. The caller must quantize Q and K
 * with info.q_scale/info.k_scale.
 * Requantization is compiled in as q_scale * k_scale / output_scale.
 */
HMX_I8_API int hmx_qk_i8_operator_matmul(
    hmx_i8_context *context,
    const int8_t *query,
    const int8_t *key,
    int8_t *score);

/* Direct-path ABI. The context owns one persistently mapped RpcMem arena.
 * Q uses HMX AH tiles [M/64,K/32,64,32], K uses HMX WH tiles
 * [N/32,K/32,8,32,4], key_sums stores one signed INT8 K sum per token, and
 * score is compact [heads,m,n]. N must be padded to 32. */
HMX_I8_API int8_t *hmx_qk_i8_operator_query_data(
    hmx_i8_context *context);
HMX_I8_API int8_t *hmx_qk_i8_operator_key_data(
    hmx_i8_context *context);
HMX_I8_API int32_t *hmx_qk_i8_operator_key_sums_data(
    hmx_i8_context *context);
HMX_I8_API const int8_t *hmx_qk_i8_operator_scores_data(
    hmx_i8_context *context);
HMX_I8_API const int32_t *hmx_qk_i8_operator_topk_indices_data(
    hmx_i8_context *context);
HMX_I8_API float *hmx_qk_i8_operator_raw_query_data(
    hmx_i8_context *context);
HMX_I8_API uint16_t *hmx_qk_i8_operator_raw_key_data(
    hmx_i8_context *context);
HMX_I8_API int hmx_qk_i8_operator_profile_raw_scales_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t query_rows,
    int32_t n,
    int32_t profile_query,
    int32_t profile_key,
    float *q_scales,
    float *k_scales);
/* Incremental K profiling scans [key_begin,n) with HVX and merges it with
 * previous_k_scales. Raw Q/K must already be staged in the context arena. */
HMX_I8_API int hmx_qk_i8_operator_profile_raw_scales_incremental_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t query_rows,
    int32_t n,
    int32_t key_begin,
    int32_t profile_query,
    const float *previous_k_scales,
    float *q_scales,
    float *k_scales);
HMX_I8_API int hmx_qk_i8_operator_last_dsp_timing(
    hmx_i8_context *context,
    hmx_i8_dsp_timing *timing);
HMX_I8_API int hmx_qk_i8_operator_prepare_raw_key_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t n,
    int32_t key_begin);
HMX_I8_API int hmx_qk_i8_operator_prepare_raw_key_scaled_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t n,
    int32_t key_begin,
    float k_scale);
HMX_I8_API int hmx_qk_i8_operator_execute_raw_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t query_rows,
    int32_t n);
/* Static-scale fast path: append/pack the K suffix and execute QK in one RPC.
 * The execution scope must already be active when one is shared by callers. */
HMX_I8_API int hmx_qk_i8_operator_prepare_execute_raw_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t query_rows,
    int32_t n,
    int32_t key_begin);
HMX_I8_API int hmx_qk_i8_operator_prepare_execute_raw_per_head_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t query_rows,
    int32_t n,
    int32_t key_begin,
    const float *q_scales,
    const float *k_scales,
    float output_scale);
/* Heterogeneous-bucket raw path with an independent accumulator-to-INT8
 * requant multiplier for every head. This keeps DSP-side Q/K profiling,
 * quantization and packing while allowing heads from different output-scale
 * buckets to share one fused RPC. */
HMX_I8_API int hmx_qk_i8_operator_prepare_execute_raw_per_head_requant_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t query_rows,
    int32_t n,
    int32_t key_begin,
    const float *q_scales,
    const float *k_scales,
    const float *requant_scales);
/* Raw-input execution with a runtime accumulator-to-INT8 requantization
 * multiplier. Q/K quantization scales remain those reported by get_info(). */
HMX_I8_API int hmx_qk_i8_operator_execute_raw_scaled_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t query_rows,
    int32_t n,
    float requant_scale);
HMX_I8_API int hmx_qk_i8_operator_execute_raw_dynamic_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t query_rows,
    int32_t n,
    float q_scale,
    float requant_scale);
/* One-head raw QK path that preserves the corrected INT32 accumulator, runs
 * exact signed Top-k on the DSP, and returns only compact ascending indices.
 * row_offsets contains query_rows + 1 cumulative offsets; it need not start
 * at zero, which allows a query chunk to use a slice of a full prompt plan. */
HMX_I8_API int hmx_qk_i8_operator_execute_raw_i32_topk_hn(
    hmx_i8_context *context,
    int32_t query_rows,
    int32_t key_len,
    int32_t n,
    int32_t causal_prefix_tokens,
    float q_scale,
    const int32_t *row_offsets,
    int32_t row_offsets_count);
HMX_I8_API int hmx_qk_i8_operator_execute_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t n);
/* Q/K and K sums are produced by one ARM worker directly in the context's
 * uncached arena. The positive ready[head] values publish input progress.
 * Once a head's output is flushed, a LONG_RPC_HEAD_READY operator replaces
 * that word with -2 while the same FastRPC invocation continues with the
 * remaining heads. -1 reports an execution failure. */
HMX_I8_API volatile int32_t *hmx_qk_i8_operator_cpu_pack_ready_data(
    hmx_i8_context *context);
HMX_I8_API int hmx_qk_i8_operator_execute_cpu_packed_per_head_hn(
    hmx_i8_context *context,
    int32_t heads,
    int32_t n,
    const float *requant_scales);
HMX_I8_API int hmx_qk_i8_operator_begin(
    hmx_i8_context *context,
    int32_t n);
HMX_I8_API int hmx_qk_i8_operator_end(hmx_i8_context *context);
HMX_I8_API int hmx_qk_i8_operator_matmul_hn(
    hmx_i8_context *context,
    const int8_t *query,
    const int8_t *key,
    int8_t *score,
    int32_t heads,
    int32_t n);

HMX_I8_API const char *hmx_i8_status_string(int status);

#ifdef __cplusplus
}
#endif

#endif /* HMX_QK_I8_OPERATOR_H */
