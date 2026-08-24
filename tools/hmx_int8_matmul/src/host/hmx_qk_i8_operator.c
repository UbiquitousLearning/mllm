#include "hmx_qk_i8_operator.h"

#include "hmx_i8_fixed.h"

#if !defined(HMX_FIXED_M) || !defined(HMX_FIXED_K) || !defined(HMX_FIXED_N) || \
    !defined(HMX_FIXED_HEADS)
#error "HMX_FIXED_HEADS/M/K/N are required"
#endif
#if !defined(HMX_OPERATOR_Q_SCALE) || !defined(HMX_OPERATOR_K_SCALE) || \
    !defined(HMX_OPERATOR_OUTPUT_SCALE)
#error "All fixed QK operator scales are required"
#endif

#define HMX_OPERATOR_REQUANT_SCALE \
    ((float)(HMX_OPERATOR_Q_SCALE) * (float)(HMX_OPERATOR_K_SCALE) / \
     (float)(HMX_OPERATOR_OUTPUT_SCALE))

int hmx_i8_matmul_qk_i8_fixed(
    hmx_i8_context *context,
    const int8_t *query,
    const int8_t *key,
    int8_t *score);
int8_t *hmx_i8_qk_query_data(hmx_i8_context *context);
int8_t *hmx_i8_qk_key_data(hmx_i8_context *context);
int32_t *hmx_i8_qk_key_sums_data(hmx_i8_context *context);
const int8_t *hmx_i8_qk_scores_data(hmx_i8_context *context);
const int32_t *hmx_i8_qk_topk_indices_data(hmx_i8_context *context);
float *hmx_i8_qk_raw_query_data(hmx_i8_context *context);
uint16_t *hmx_i8_qk_raw_key_data(hmx_i8_context *context);
int hmx_i8_prepare_qk_i8_raw_key(
    hmx_i8_context *context, int32_t heads, int32_t n, int32_t key_begin);
int hmx_i8_prepare_qk_i8_raw_key_scaled(
    hmx_i8_context *context, int32_t heads, int32_t n, int32_t key_begin,
    float k_scale);
int hmx_i8_profile_qk_raw_scales(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t profile_query, int32_t profile_key, float *q_scales,
    float *k_scales);
int hmx_i8_profile_qk_raw_scales_incremental(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, int32_t profile_query,
    const float *previous_k_scales, float *q_scales, float *k_scales);
int hmx_i8_last_dsp_timing(
    hmx_i8_context *context, hmx_i8_dsp_timing *timing);
int hmx_i8_execute_qk_i8_raw(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n);
int hmx_i8_prepare_execute_qk_i8_raw(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin);
int hmx_i8_prepare_execute_qk_i8_raw_per_head(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, const float *q_scales, const float *k_scales,
    float output_scale);
int hmx_i8_prepare_execute_qk_i8_raw_per_head_requant(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, const float *q_scales, const float *k_scales,
    const float *requant_scales);
int hmx_i8_execute_qk_i8_raw_scaled(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    float requant_scale);
int hmx_i8_execute_qk_i8_raw_dynamic(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    float q_scale, float requant_scale);
int hmx_i8_execute_qk_i32_topk_raw(
    hmx_i8_context *context, int32_t query_rows, int32_t key_len, int32_t n,
    int32_t causal_prefix_tokens, float q_scale,
    const int32_t *row_offsets, int32_t row_offsets_count);
int hmx_i8_execute_qk_i8_direct(
    hmx_i8_context *context, int32_t heads, int32_t n);
volatile int32_t *hmx_i8_qk_cpu_pack_ready_data(hmx_i8_context *context);
int hmx_i8_execute_qk_i8_cpu_packed_per_head(
    hmx_i8_context *context, int32_t heads, int32_t n,
    const float *requant_scales);
int hmx_i8_begin_qk_i8_packed(hmx_i8_context *context, int32_t n);
int hmx_i8_end_qk_i8_packed(hmx_i8_context *context);
int hmx_i8_matmul_qk_i8_direct(
    hmx_i8_context *context, const int8_t *query, const int8_t *key,
    int8_t *score, int32_t heads, int32_t n);

uint32_t hmx_qk_i8_operator_api_version(void) {
    return HMX_QK_I8_OPERATOR_API_VERSION;
}

int hmx_qk_i8_operator_get_info(hmx_qk_i8_operator_info *info) {
    if (!info) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    *info = (hmx_qk_i8_operator_info){
        .struct_size = sizeof(*info),
        .heads = HMX_FIXED_HEADS,
        .m = HMX_FIXED_M,
        .k = HMX_FIXED_K,
        .n = HMX_FIXED_N,
        .q_scale = (float)(HMX_OPERATOR_Q_SCALE),
        .k_scale = (float)(HMX_OPERATOR_K_SCALE),
        .output_scale = (float)(HMX_OPERATOR_OUTPUT_SCALE),
        .requant_scale = HMX_OPERATOR_REQUANT_SCALE,
        .flags = HMX_QK_I8_FLAG_PERSISTENT_RPCMEM |
                 HMX_QK_I8_FLAG_DIRECT_IO |
                 HMX_QK_I8_FLAG_N_IS_CAPACITY |
                 HMX_QK_I8_FLAG_HEADS_IS_CAPACITY |
                 HMX_QK_I8_FLAG_DSP_LAYOUT |
                 HMX_QK_I8_FLAG_PREPACKED_QK |
                 HMX_QK_I8_FLAG_KEY_SUMS |
                 HMX_QK_I8_FLAG_EXECUTION_SCOPE |
                 HMX_QK_I8_FLAG_DMA_PIPELINE |
                 HMX_QK_I8_FLAG_RAW_QK_HVX |
                 HMX_QK_I8_FLAG_INCREMENTAL_K |
                 HMX_QK_I8_FLAG_HVX_SCALE_PROFILE |
                 HMX_QK_I8_FLAG_PERSISTENT_DSP_ARENA |
                 HMX_QK_I8_FLAG_FUSED_RAW_QK |
                 HMX_QK_I8_FLAG_PER_HEAD_BUCKET_SCALES |
                 HMX_QK_I8_FLAG_CPU_PACKED_PIPELINE |
                 HMX_QK_I8_FLAG_LONG_RPC_HEAD_READY,
    };
    return HMX_I8_OK;
}

int8_t *hmx_qk_i8_operator_query_data(hmx_i8_context *context) {
    return hmx_i8_qk_query_data(context);
}

int8_t *hmx_qk_i8_operator_key_data(hmx_i8_context *context) {
    return hmx_i8_qk_key_data(context);
}

int32_t *hmx_qk_i8_operator_key_sums_data(hmx_i8_context *context) {
    return hmx_i8_qk_key_sums_data(context);
}

const int8_t *hmx_qk_i8_operator_scores_data(hmx_i8_context *context) {
    return hmx_i8_qk_scores_data(context);
}

const int32_t *hmx_qk_i8_operator_topk_indices_data(
    hmx_i8_context *context) {
    return hmx_i8_qk_topk_indices_data(context);
}

float *hmx_qk_i8_operator_raw_query_data(hmx_i8_context *context) {
    return hmx_i8_qk_raw_query_data(context);
}

uint16_t *hmx_qk_i8_operator_raw_key_data(hmx_i8_context *context) {
    return hmx_i8_qk_raw_key_data(context);
}

int hmx_qk_i8_operator_profile_raw_scales_hn(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t profile_query, int32_t profile_key, float *q_scales,
    float *k_scales) {
    return hmx_i8_profile_qk_raw_scales(
        context, heads, query_rows, n, profile_query, profile_key,
        q_scales, k_scales);
}

int hmx_qk_i8_operator_profile_raw_scales_incremental_hn(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, int32_t profile_query,
    const float *previous_k_scales, float *q_scales, float *k_scales) {
    return hmx_i8_profile_qk_raw_scales_incremental(
        context, heads, query_rows, n, key_begin, profile_query,
        previous_k_scales, q_scales, k_scales);
}

int hmx_qk_i8_operator_last_dsp_timing(
    hmx_i8_context *context, hmx_i8_dsp_timing *timing) {
    return hmx_i8_last_dsp_timing(context, timing);
}

int hmx_qk_i8_operator_prepare_raw_key_hn(
    hmx_i8_context *context, int32_t heads, int32_t n, int32_t key_begin) {
    return hmx_i8_prepare_qk_i8_raw_key(context, heads, n, key_begin);
}

int hmx_qk_i8_operator_prepare_raw_key_scaled_hn(
    hmx_i8_context *context, int32_t heads, int32_t n, int32_t key_begin,
    float k_scale) {
    return hmx_i8_prepare_qk_i8_raw_key_scaled(
        context, heads, n, key_begin, k_scale);
}

int hmx_qk_i8_operator_execute_raw_hn(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n) {
    return hmx_i8_execute_qk_i8_raw(context, heads, query_rows, n);
}

int hmx_qk_i8_operator_prepare_execute_raw_hn(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin) {
    return hmx_i8_prepare_execute_qk_i8_raw(
        context, heads, query_rows, n, key_begin);
}

int hmx_qk_i8_operator_prepare_execute_raw_per_head_hn(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, const float *q_scales, const float *k_scales,
    float output_scale) {
    return hmx_i8_prepare_execute_qk_i8_raw_per_head(
        context, heads, query_rows, n, key_begin, q_scales, k_scales,
        output_scale);
}

int hmx_qk_i8_operator_prepare_execute_raw_per_head_requant_hn(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, const float *q_scales, const float *k_scales,
    const float *requant_scales) {
    return hmx_i8_prepare_execute_qk_i8_raw_per_head_requant(
        context, heads, query_rows, n, key_begin, q_scales, k_scales,
        requant_scales);
}

int hmx_qk_i8_operator_execute_raw_scaled_hn(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    float requant_scale) {
    return hmx_i8_execute_qk_i8_raw_scaled(
        context, heads, query_rows, n, requant_scale);
}

int hmx_qk_i8_operator_execute_raw_dynamic_hn(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    float q_scale, float requant_scale) {
    return hmx_i8_execute_qk_i8_raw_dynamic(
        context, heads, query_rows, n, q_scale, requant_scale);
}

int hmx_qk_i8_operator_execute_raw_i32_topk_hn(
    hmx_i8_context *context, int32_t query_rows, int32_t key_len, int32_t n,
    int32_t causal_prefix_tokens, float q_scale,
    const int32_t *row_offsets, int32_t row_offsets_count) {
    return hmx_i8_execute_qk_i32_topk_raw(
        context, query_rows, key_len, n, causal_prefix_tokens, q_scale,
        row_offsets, row_offsets_count);
}

int hmx_qk_i8_operator_execute_hn(
    hmx_i8_context *context, int32_t heads, int32_t n) {
    return hmx_i8_execute_qk_i8_direct(context, heads, n);
}

volatile int32_t *hmx_qk_i8_operator_cpu_pack_ready_data(
    hmx_i8_context *context) {
    return hmx_i8_qk_cpu_pack_ready_data(context);
}

int hmx_qk_i8_operator_execute_cpu_packed_per_head_hn(
    hmx_i8_context *context, int32_t heads, int32_t n,
    const float *requant_scales) {
    return hmx_i8_execute_qk_i8_cpu_packed_per_head(
        context, heads, n, requant_scales);
}

int hmx_qk_i8_operator_begin(hmx_i8_context *context, int32_t n) {
    return hmx_i8_begin_qk_i8_packed(context, n);
}

int hmx_qk_i8_operator_end(hmx_i8_context *context) {
    return hmx_i8_end_qk_i8_packed(context);
}

int hmx_qk_i8_operator_matmul_hn(
    hmx_i8_context *context, const int8_t *query, const int8_t *key,
    int8_t *score, int32_t heads, int32_t n) {
    return hmx_i8_matmul_qk_i8_direct(
        context, query, key, score, heads, n);
}

int hmx_qk_i8_operator_matmul(
    hmx_i8_context *context,
    const int8_t *query,
    const int8_t *key,
    int8_t *score) {
    return hmx_i8_matmul_qk_i8_fixed(
        context,
        query,
        key,
        score);
}
