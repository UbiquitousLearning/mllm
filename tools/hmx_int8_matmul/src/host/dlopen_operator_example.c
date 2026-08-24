#include "hmx_qk_i8_operator.h"

#include <dlfcn.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint32_t (*operator_api_version_fn)(void);
typedef int (*operator_get_info_fn)(hmx_qk_i8_operator_info *);
typedef int (*create_fn)(hmx_i8_context **);
typedef void (*destroy_fn)(hmx_i8_context *);
typedef int (*operator_matmul_fn)(
    hmx_i8_context *, const int8_t *, const int8_t *, int8_t *);
typedef int8_t *(*operator_data_fn)(hmx_i8_context *);
typedef int32_t *(*operator_i32_data_fn)(hmx_i8_context *);
typedef const int8_t *(*operator_const_data_fn)(hmx_i8_context *);
typedef float *(*operator_raw_query_data_fn)(hmx_i8_context *);
typedef uint16_t *(*operator_raw_key_data_fn)(hmx_i8_context *);
typedef int (*operator_execute_hn_fn)(hmx_i8_context *, int32_t, int32_t);
typedef int (*operator_prepare_raw_key_fn)(
    hmx_i8_context *, int32_t, int32_t, int32_t);
typedef int (*operator_profile_raw_fn)(
    hmx_i8_context *, int32_t, int32_t, int32_t, int32_t, int32_t,
    float *, float *);
typedef int (*operator_execute_raw_hn_fn)(
    hmx_i8_context *, int32_t, int32_t, int32_t);
typedef int (*operator_prepare_execute_raw_hn_fn)(
    hmx_i8_context *, int32_t, int32_t, int32_t, int32_t);
typedef int (*operator_prepare_execute_raw_per_head_hn_fn)(
    hmx_i8_context *, int32_t, int32_t, int32_t, int32_t,
    const float *, const float *, float);
typedef int (*operator_begin_fn)(hmx_i8_context *, int32_t);
typedef int (*operator_end_fn)(hmx_i8_context *);
typedef const char *(*status_string_fn)(int);

typedef struct operator_api {
    operator_api_version_fn api_version;
    operator_get_info_fn get_info;
    create_fn create;
    destroy_fn destroy;
    operator_matmul_fn matmul;
    operator_data_fn query_data;
    operator_data_fn key_data;
    operator_i32_data_fn key_sums_data;
    operator_const_data_fn scores_data;
    operator_raw_query_data_fn raw_query_data;
    operator_raw_key_data_fn raw_key_data;
    operator_execute_hn_fn execute_hn;
    operator_prepare_raw_key_fn prepare_raw_key;
    operator_profile_raw_fn profile_raw;
    operator_execute_raw_hn_fn execute_raw_hn;
    operator_prepare_execute_raw_hn_fn prepare_execute_raw_hn;
    operator_prepare_execute_raw_per_head_hn_fn
        prepare_execute_raw_per_head_hn;
    operator_begin_fn begin;
    operator_end_fn end;
    status_string_fn status_string;
} operator_api;

static int load_symbol(void *library, const char *name, void **output) {
    dlerror();
    void *symbol = dlsym(library, name);
    const char *error = dlerror();
    if (error) {
        fprintf(stderr, "dlsym(%s): %s\n", name, error);
        return -1;
    }
    *output = symbol;
    return 0;
}

static int load_api(void *library, operator_api *api) {
    return load_symbol(
               library,
               "hmx_qk_i8_operator_api_version",
               (void **)&api->api_version) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_get_info",
            (void **)&api->get_info) ||
        load_symbol(library, "hmx_i8_create", (void **)&api->create) ||
        load_symbol(library, "hmx_i8_destroy", (void **)&api->destroy) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_matmul",
            (void **)&api->matmul) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_query_data",
            (void **)&api->query_data) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_key_data",
            (void **)&api->key_data) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_key_sums_data",
            (void **)&api->key_sums_data) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_scores_data",
            (void **)&api->scores_data) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_raw_query_data",
            (void **)&api->raw_query_data) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_raw_key_data",
            (void **)&api->raw_key_data) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_execute_hn",
            (void **)&api->execute_hn) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_prepare_raw_key_hn",
            (void **)&api->prepare_raw_key) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_profile_raw_scales_hn",
            (void **)&api->profile_raw) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_execute_raw_hn",
            (void **)&api->execute_raw_hn) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_prepare_execute_raw_hn",
            (void **)&api->prepare_execute_raw_hn) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_prepare_execute_raw_per_head_hn",
            (void **)&api->prepare_execute_raw_per_head_hn) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_begin",
            (void **)&api->begin) ||
        load_symbol(
            library,
            "hmx_qk_i8_operator_end",
            (void **)&api->end) ||
        load_symbol(
            library,
            "hmx_i8_status_string",
            (void **)&api->status_string);
}

static int verify_runtime_entry_hidden(void *library) {
    dlerror();
    void *symbol = dlsym(library, "hmx_i8_matmul_i8_fixed");
    const char *error = dlerror();
    if (symbol || !error) {
        fprintf(stderr, "runtime-scale entry is unexpectedly exported\n");
        return -1;
    }
    return 0;
}

static int checked_count(int32_t rows, int32_t columns, size_t *count) {
    if (rows <= 0 || columns <= 0 || (size_t)rows > SIZE_MAX / (size_t)columns) {
        return -1;
    }
    *count = (size_t)rows * (size_t)columns;
    return 0;
}

static int8_t requantize_reference(int32_t accumulator, float scale) {
    float scaled = (float)accumulator * scale;
    if (scaled >= 127.0f) {
        return INT8_MAX;
    }
    if (scaled <= -128.0f) {
        return INT8_MIN;
    }
    return (int8_t)(int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
}

static int validate_info(const hmx_qk_i8_operator_info *info) {
    float expected_requant =
        info->q_scale * info->k_scale / info->output_scale;
    float tolerance = 1.0e-6f * fmaxf(1.0f, fabsf(expected_requant));
    return info->struct_size == sizeof(*info) &&
            info->heads > 0 && info->m > 0 && info->k > 0 && info->n > 0 &&
            isfinite(info->q_scale) && info->q_scale > 0.0f &&
            isfinite(info->k_scale) && info->k_scale > 0.0f &&
            isfinite(info->output_scale) && info->output_scale > 0.0f &&
            isfinite(info->requant_scale) &&
            fabsf(info->requant_scale - expected_requant) <= tolerance
        ? 0
        : -1;
}

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0.0;
    }
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

typedef struct ranked_score {
    int32_t score;
    int32_t index;
} ranked_score;

typedef enum input_pattern {
    INPUT_FULL_RANGE,
    INPUT_REALISTIC,
} input_pattern;

static uint32_t next_random(uint32_t *state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static int8_t make_input_value(uint32_t *state, size_t index,
                               input_pattern pattern) {
    const uint32_t random = next_random(state);
    if (pattern == INPUT_FULL_RANGE) {
        return (int8_t)((int)(random % 255u) - 127);
    }

    /* A centered, approximately triangular distribution with rare extrema.
     * This resembles a max-calibrated activation tensor: most quantized
     * values are far from +/-127, but the calibration range is exercised. */
    if (index % 4093u == 0u) {
        return (index / 4093u) & 1u ? INT8_C(127) : INT8_C(-127);
    }
    int value = (int)(random & 255u) - (int)((random >> 8) & 255u);
    value /= 5;
    if (value < -127) value = -127;
    if (value > 127) value = 127;
    return (int8_t)value;
}

static uint16_t float_to_half_bits(float value) {
    const _Float16 half = (_Float16)value;
    uint16_t bits = 0;
    memcpy(&bits, &half, sizeof(bits));
    return bits;
}

static float half_bits_to_float(uint16_t bits) {
    _Float16 half;
    memcpy(&half, &bits, sizeof(half));
    return (float)half;
}

static int compare_ranked_score(const void *left, const void *right) {
    const ranked_score *a = (const ranked_score *)left;
    const ranked_score *b = (const ranked_score *)right;
    if (a->score != b->score) return a->score > b->score ? -1 : 1;
    if (a->index != b->index) return a->index < b->index ? -1 : 1;
    return 0;
}

static int verify_case(
    const operator_api *api,
    hmx_i8_context *context,
    const hmx_qk_i8_operator_info *info,
    const char *name,
    int32_t heads,
    int32_t n,
    input_pattern pattern,
    int benchmark_iterations) {
    size_t query_per_head;
    size_t key_per_head;
    size_t score_per_head;
    if (heads <= 0 || heads > info->heads || n < info->m || n > info->n ||
        (n % 32) != 0 ||
        checked_count(info->m, info->k, &query_per_head) ||
        checked_count(n, info->k, &key_per_head) ||
        checked_count(info->m, n, &score_per_head) ||
        (size_t)heads > SIZE_MAX / query_per_head ||
        (size_t)heads > SIZE_MAX / key_per_head ||
        (size_t)heads > SIZE_MAX / score_per_head) {
        return -1;
    }
    const size_t query_count = (size_t)heads * query_per_head;
    const size_t key_count = (size_t)heads * key_per_head;
    const size_t score_count = (size_t)heads * score_per_head;
    int8_t *query = api->query_data(context);
    int8_t *key = api->key_data(context);
    int32_t *key_sums = api->key_sums_data(context);
    const int8_t *score = api->scores_data(context);
    int32_t *dense_row = (int32_t *)malloc((size_t)n * sizeof(*dense_row));
    ranked_score *dense_rank =
        (ranked_score *)malloc((size_t)n * sizeof(*dense_rank));
    ranked_score *hmx_rank =
        (ranked_score *)malloc((size_t)n * sizeof(*hmx_rank));
    uint8_t *selected = (uint8_t *)malloc((size_t)n);
    if (!query || !key || !key_sums || !score || !dense_row || !dense_rank ||
        !hmx_rank || !selected) {
        free(dense_row);
        free(dense_rank);
        free(hmx_rank);
        free(selected);
        return -1;
    }

    uint32_t query_state = UINT32_C(0x9e3779b9) ^ (uint32_t)n ^
        ((uint32_t)heads << 16);
    uint32_t key_state = UINT32_C(0x243f6a88) ^ (uint32_t)n ^
        ((uint32_t)heads << 20);
    int8_t *query_row_major = (int8_t *)malloc(query_count);
    int8_t *key_row_major = (int8_t *)malloc(key_count);
    if (!query_row_major || !key_row_major) return -1;
    memset(query, 0, query_count);
    memset(key, 0, key_count);
    memset(key_sums, 0, (size_t)heads * (size_t)n * sizeof(*key_sums));
    const int k_tiles = info->k / 32;
    for (size_t index = 0; index < query_count; ++index) {
        query_row_major[index] = make_input_value(&query_state, index, pattern);
    }
    for (size_t index = 0; index < key_count; ++index) {
        key_row_major[index] = make_input_value(&key_state, index + 17u, pattern);
    }
    for (int32_t head = 0; head < heads; ++head) {
        uint8_t *packed_q = (uint8_t *)query + (size_t)head * query_per_head;
        int8_t *packed_k = key + (size_t)head * key_per_head;
        int32_t *sums = key_sums + (size_t)head * (size_t)n;
        for (int32_t row = 0; row < info->m; ++row) {
            for (int32_t inner = 0; inner < info->k; ++inner) {
                const int8_t value = query_row_major[
                    ((size_t)head * info->m + row) * info->k + inner];
                const size_t packed =
                    ((size_t)(row / 64) * k_tiles + (size_t)(inner / 32))
                        * 2048u + (size_t)(row % 64) * 32u
                    + (size_t)(inner % 32);
                packed_q[packed] = (uint8_t)value ^ UINT8_C(0x80);
            }
        }
        for (int32_t token = 0; token < n; ++token) {
            int32_t sum = 0;
            for (int32_t inner = 0; inner < info->k; ++inner) {
                const int8_t value = key_row_major[
                    ((size_t)head * n + token) * info->k + inner];
                sum += value;
                const size_t packed =
                    ((size_t)(token / 32) * k_tiles + (size_t)(inner / 32))
                        * 1024u
                    + (size_t)((inner % 32) / 4) * 128u
                    + (size_t)(token % 32) * 4u + (size_t)(inner % 4);
                packed_k[packed] = value;
            }
            sums[token] = sum;
        }
    }

    int status = api->begin(context, n);
    const int scope_started = status == HMX_I8_OK;
    if (scope_started) status = api->execute_hn(context, heads, n);
    if (scope_started && api->end(context) != HMX_I8_OK &&
        status == HMX_I8_OK) status = -1;
    if (status != HMX_I8_OK) {
        fprintf(stderr, "%s direct operator call failed: %s (%d)\n",
                name, api->status_string(status), status);
        free(dense_row);
        free(dense_rank);
        free(hmx_rank);
        free(selected);
        return -1;
    }

    int8_t *packed_reference = (int8_t *)malloc(score_count);
    float *raw_query = api->raw_query_data(context);
    uint16_t *raw_key = api->raw_key_data(context);
    if (!packed_reference || !raw_query || !raw_key) return -1;
    memcpy(packed_reference, score, score_count);
    for (size_t index = 0; index < query_count; ++index) {
        raw_query[index] = (float)query_row_major[index] * info->q_scale;
    }
    for (size_t index = 0; index < key_count; ++index) {
        raw_key[index] = float_to_half_bits(
            (float)key_row_major[index] * info->k_scale);
    }
    float *profile_q = (float *)calloc((size_t)heads, sizeof(float));
    float *profile_k = (float *)calloc((size_t)heads, sizeof(float));
    size_t profile_mismatches = 0;
    if (!profile_q || !profile_k ||
        api->profile_raw(context, heads, info->m, n, 1, 1,
                         profile_q, profile_k) != HMX_I8_OK) {
        fprintf(stderr, "%s raw HVX scale profile failed\n", name);
        free(profile_q);
        free(profile_k);
        return -1;
    }
    for (int32_t head = 0; head < heads; ++head) {
        float q_maximum = 0.0f;
        float k_maximum = 0.0f;
        for (size_t index = 0; index < query_per_head; ++index) {
            q_maximum = fmaxf(
                q_maximum,
                fabsf(raw_query[(size_t)head * query_per_head + index]));
        }
        for (size_t index = 0; index < key_per_head; ++index) {
            k_maximum = fmaxf(
                k_maximum,
                fabsf(half_bits_to_float(
                    raw_key[(size_t)head * key_per_head + index])));
        }
        const float expected_q = q_maximum / 127.0f;
        const float expected_k = k_maximum / 127.0f;
        profile_mismatches += fabsf(profile_q[head] - expected_q) >
            fmaxf(1.0e-7f, expected_q * 1.0e-5f);
        profile_mismatches += fabsf(profile_k[head] - expected_k) >
            fmaxf(1.0e-7f, expected_k * 1.0e-5f);
    }
    printf("RAW_HVX_SCALE_RESULT name=%s status=%s mismatches=%zu\n",
           name, profile_mismatches == 0 ? "PASS" : "FAIL",
           profile_mismatches);
    free(profile_q);
    free(profile_k);
    if (profile_mismatches != 0) return -1;
    status = api->begin(context, n);
    const int raw_scope_started = status == HMX_I8_OK;
    if (raw_scope_started) {
        status = api->prepare_raw_key(context, heads, n, 0);
    }
    if (status == HMX_I8_OK) {
        status = api->execute_raw_hn(context, heads, info->m, n);
    }
    if (raw_scope_started && api->end(context) != HMX_I8_OK &&
        status == HMX_I8_OK) status = -1;
    size_t raw_mismatches = 0;
    size_t raw_q_layout_mismatches = 0;
    size_t raw_k_layout_mismatches = 0;
    size_t raw_sum_mismatches = 0;
    if (status == HMX_I8_OK) {
        for (size_t index = 0; index < score_count; ++index) {
            raw_mismatches += score[index] != packed_reference[index];
        }
        for (int32_t head = 0; head < heads; ++head) {
            for (int32_t row = 0; row < info->m; ++row) {
                for (int32_t inner = 0; inner < info->k; ++inner) {
                    const size_t packed = (size_t)head * query_per_head
                        + ((size_t)(row / 64) * k_tiles
                           + (size_t)(inner / 32)) * 2048u
                        + (size_t)(row % 64) * 32u + (size_t)(inner % 32);
                    const uint8_t expected = (uint8_t)query_row_major[
                        ((size_t)head * info->m + row) * info->k + inner]
                        ^ UINT8_C(0x80);
                    if (((const uint8_t *)query)[packed] != expected &&
                        raw_q_layout_mismatches < 12) {
                        fprintf(stderr,
                                "raw Q mismatch h=%d r=%d k=%d got=%d expected=%d raw=%.9g inv=%.9g\n",
                                head, row, inner,
                                (int)((const uint8_t *)query)[packed] - 128,
                                (int)query_row_major[
                                    ((size_t)head * info->m + row)
                                    * info->k + inner],
                                raw_query[((size_t)head * info->m + row)
                                          * info->k + inner],
                                1.0f / info->q_scale);
                    }
                    raw_q_layout_mismatches +=
                        ((const uint8_t *)query)[packed] != expected;
                }
            }
            for (int32_t token = 0; token < n; ++token) {
                int32_t expected_sum = 0;
                for (int32_t inner = 0; inner < info->k; ++inner) {
                    const int8_t expected = key_row_major[
                        ((size_t)head * n + token) * info->k + inner];
                    expected_sum += expected;
                    const size_t packed = (size_t)head * key_per_head
                        + ((size_t)(token / 32) * k_tiles
                           + (size_t)(inner / 32)) * 1024u
                        + (size_t)((inner % 32) / 4) * 128u
                        + (size_t)(token % 32) * 4u + (size_t)(inner % 4);
                    if (key[packed] != expected &&
                        raw_k_layout_mismatches < 12) {
                        fprintf(stderr,
                                "raw K mismatch h=%d t=%d k=%d got=%d expected=%d\n",
                                head, token, inner, (int)key[packed],
                                (int)expected);
                    }
                    raw_k_layout_mismatches += key[packed] != expected;
                }
                raw_sum_mismatches += key_sums[
                    (size_t)head * (size_t)n + token] != expected_sum;
            }
        }
    }
    printf("RAW_HVX_RESULT name=%s status=%s mismatches=%zu elements=%zu "
           "q_layout_mismatches=%zu k_layout_mismatches=%zu "
           "sum_mismatches=%zu\n",
           name,
           status == HMX_I8_OK && raw_mismatches == 0 ? "PASS" : "FAIL",
           raw_mismatches, score_count, raw_q_layout_mismatches,
           raw_k_layout_mismatches, raw_sum_mismatches);
    if (status != HMX_I8_OK || raw_mismatches != 0) return -1;

    status = api->begin(context, n);
    const int fused_scope_started = status == HMX_I8_OK;
    if (fused_scope_started) {
        status = api->prepare_execute_raw_hn(
            context, heads, info->m, n, 0);
    }
    if (fused_scope_started && api->end(context) != HMX_I8_OK &&
        status == HMX_I8_OK) status = -1;
    size_t fused_mismatches = 0;
    if (status == HMX_I8_OK) {
        for (size_t index = 0; index < score_count; ++index) {
            fused_mismatches += score[index] != packed_reference[index];
        }
    }
    printf("FUSED_RAW_HVX_RESULT name=%s status=%s mismatches=%zu "
           "elements=%zu\n",
           name,
           status == HMX_I8_OK && fused_mismatches == 0 ? "PASS" : "FAIL",
           fused_mismatches, score_count);
    if (status != HMX_I8_OK || fused_mismatches != 0) return -1;

    if (heads > 1 && n <= 256) {
        float *head_q_scales = (float *)malloc(
            (size_t)heads * sizeof(*head_q_scales));
        float *head_k_scales = (float *)malloc(
            (size_t)heads * sizeof(*head_k_scales));
        if (!head_q_scales || !head_k_scales) return -1;
        for (int32_t head = 0; head < heads; ++head) {
            const float q_multiplier = head % 3 == 0 ? 0.5f
                : (head % 3 == 1 ? 1.0f : 2.0f);
            const float k_multiplier = head % 3 == 0 ? 2.0f
                : (head % 3 == 1 ? 0.5f : 1.0f);
            head_q_scales[head] = info->q_scale * q_multiplier;
            head_k_scales[head] = info->k_scale * k_multiplier;
            for (size_t index = 0; index < query_per_head; ++index) {
                raw_query[(size_t)head * query_per_head + index] =
                    (float)query_row_major[
                        (size_t)head * query_per_head + index]
                    * head_q_scales[head];
            }
            for (size_t index = 0; index < key_per_head; ++index) {
                raw_key[(size_t)head * key_per_head + index] =
                    float_to_half_bits(
                        (float)key_row_major[
                            (size_t)head * key_per_head + index]
                        * head_k_scales[head]);
            }
        }
        status = api->begin(context, n);
        const int per_head_scope_started = status == HMX_I8_OK;
        if (per_head_scope_started) {
            status = api->prepare_execute_raw_per_head_hn(
                context, heads, info->m, n, 0, head_q_scales,
                head_k_scales, info->output_scale);
        }
        if (per_head_scope_started && api->end(context) != HMX_I8_OK &&
            status == HMX_I8_OK) status = -1;
        size_t per_head_mismatches = 0;
        if (status == HMX_I8_OK) {
            for (int32_t head = 0; head < heads; ++head) {
                const float requant = head_q_scales[head]
                    * head_k_scales[head] / info->output_scale;
                for (int32_t row = 0; row < info->m; ++row) {
                    for (int32_t column = 0; column < n; ++column) {
                        int32_t accumulator = 0;
                        for (int32_t inner = 0; inner < info->k; ++inner) {
                            accumulator += (int32_t)query_row_major[
                                ((size_t)head * info->m + row) * info->k
                                + inner] * (int32_t)key_row_major[
                                ((size_t)head * n + column) * info->k
                                + inner];
                        }
                        const int8_t expected = requantize_reference(
                            accumulator, requant);
                        const int8_t actual = score[
                            ((size_t)head * info->m + row) * n + column];
                        per_head_mismatches += actual != expected;
                    }
                }
            }
        }
        printf("PER_HEAD_BUCKET_FUSION_RESULT name=%s status=%s "
               "mismatches=%zu elements=%zu\n",
               name,
               status == HMX_I8_OK && per_head_mismatches == 0
                   ? "PASS" : "FAIL",
               per_head_mismatches, score_count);
        free(head_q_scales);
        free(head_k_scales);
        if (status != HMX_I8_OK || per_head_mismatches != 0) return -1;
    }

    if (benchmark_iterations > 0) {
        const double begin = monotonic_seconds();
        for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
            status = api->begin(context, n);
            if (status == HMX_I8_OK) {
                status = api->execute_hn(context, heads, n);
                const int end_status = api->end(context);
                if (status == HMX_I8_OK) status = end_status;
            }
            if (status != HMX_I8_OK) {
                fprintf(stderr, "%s benchmark failed: %s (%d)\n", name,
                        api->status_string(status), status);
                free(dense_row);
                free(dense_rank);
                free(hmx_rank);
                free(selected);
                return -1;
            }
        }
        const double elapsed = monotonic_seconds() - begin;
        if (elapsed > 0.0) {
            const double average_ms = elapsed * 1000.0 /
                (double)benchmark_iterations;
            const double operations = 2.0 * (double)heads * (double)info->m *
                (double)info->k * (double)n;
            const double tops = operations /
                (elapsed / (double)benchmark_iterations) / 1.0e12;
            printf("CASE_BENCH name=%s iterations=%d average_ms=%.3f "
                   "effective_TOPS=%.3f\n", name, benchmark_iterations,
                   average_ms, tops);
        }
    }

    size_t mismatches = 0;
    size_t zero_count = 0;
    size_t saturated_count = 0;
    uint8_t observed[256] = {0};
    uint64_t retained_total = 0;
    uint64_t overlap_total = 0;
    double jaccard_sum = 0.0;
    uint64_t row_total = 0;
    const int tile_rows = info->m / 64;
    const int tile_columns = n / 32;
    size_t *tile_mismatches = (size_t *)calloc(
        (size_t)tile_rows * tile_columns, sizeof(*tile_mismatches));
    size_t *tile_zeros = (size_t *)calloc(
        (size_t)tile_rows * tile_columns, sizeof(*tile_zeros));
    if (!tile_mismatches || !tile_zeros) {
        free(tile_mismatches);
        free(tile_zeros);
        free(dense_row);
        free(dense_rank);
        free(hmx_rank);
        free(selected);
        return -1;
    }
    for (int32_t head = 0; head < heads; ++head) {
        const int8_t *head_query =
            query_row_major + (size_t)head * query_per_head;
        const int8_t *head_key =
            key_row_major + (size_t)head * key_per_head;
        const int8_t *head_score = packed_reference
            + (size_t)head * score_per_head;
        for (int32_t row = 0; row < info->m; ++row) {
            for (int32_t column = 0; column < n; ++column) {
                int32_t accumulator = 0;
                for (int32_t inner = 0; inner < info->k; ++inner) {
                    accumulator +=
                        (int32_t)head_query[
                            (size_t)row * (size_t)info->k + inner] *
                        (int32_t)head_key[
                            (size_t)column * (size_t)info->k + inner];
                }
                int8_t expected = requantize_reference(
                    accumulator,
                    info->requant_scale);
                int8_t actual =
                    head_score[(size_t)row * (size_t)n + column];
                dense_row[column] = accumulator;
                zero_count += actual == 0;
                saturated_count += actual == INT8_MIN || actual == INT8_MAX;
                observed[(unsigned int)((int)actual + 128)] = 1;
                if (head == 0) {
                    const size_t tile = (size_t)(row / 64) * tile_columns +
                        (size_t)(column / 32);
                    tile_zeros[tile] += actual == 0;
                    tile_mismatches[tile] += actual != expected;
                }
                if (actual != expected) {
                    if (mismatches < 8u) {
                        fprintf(
                            stderr,
                            "mismatch [h=%d,%d,%d]: DSP=%d CPU=%d\n",
                            head,
                            row,
                            column,
                            (int)actual,
                            (int)expected);
                    }
                    ++mismatches;
                }
            }

            /* Match the causal chunk shape used by prefill and compare the
             * retained 20% against the pre-requantized dense integer QK.
             * This isolates information lost by the INT8 score tensor. */
            const int valid = n - info->m + row + 1;
            const int keep = (valid + 4) / 5;
            memset(selected, 0, (size_t)valid);
            for (int32_t column = 0; column < valid; ++column) {
                dense_rank[column] = (ranked_score){
                    .score = dense_row[column], .index = column};
                hmx_rank[column] = (ranked_score){
                    .score = head_score[(size_t)row * (size_t)n + column],
                    .index = column};
            }
            qsort(dense_rank, (size_t)valid, sizeof(*dense_rank),
                  compare_ranked_score);
            qsort(hmx_rank, (size_t)valid, sizeof(*hmx_rank),
                  compare_ranked_score);
            for (int32_t rank = 0; rank < keep; ++rank) {
                selected[dense_rank[rank].index] = 1;
            }
            uint64_t row_overlap = 0;
            for (int32_t rank = 0; rank < keep; ++rank) {
                row_overlap += selected[hmx_rank[rank].index] != 0;
            }
            retained_total += (uint64_t)keep;
            overlap_total += row_overlap;
            jaccard_sum += (double)row_overlap /
                (double)(2 * keep - (int32_t)row_overlap);
            ++row_total;
        }
    }

    int unique_outputs = 0;
    for (size_t index = 0; index < sizeof(observed); ++index) {
        unique_outputs += observed[index] != 0;
    }
    const double topk_recall = retained_total == 0 ? 0.0 :
        (double)overlap_total / (double)retained_total;
    const double mean_jaccard = row_total == 0 ? 0.0 :
        jaccard_sum / (double)row_total;
    const int nondegenerate = unique_outputs > 1 && zero_count < score_count;

    printf(
        "CASE_RESULT name=%s status=%s mismatches=%zu elements=%zu "
        "H=%d M=%d K=%d N=%d pattern=%s unique_outputs=%d "
        "zero_fraction=%.6f saturation_fraction=%.6f "
        "topk20_recall=%.6f topk20_jaccard=%.6f\n",
        name,
        mismatches == 0 && nondegenerate ? "PASS" : "FAIL",
        mismatches,
        score_count,
        heads,
        info->m,
        info->k,
        n,
        pattern == INPUT_FULL_RANGE ? "full-range" : "realistic",
        unique_outputs,
        (double)zero_count / (double)score_count,
        (double)saturated_count / (double)score_count,
        topk_recall,
        mean_jaccard);
    if (n <= 256) {
        for (int tile_row = 0; tile_row < tile_rows; ++tile_row) {
            for (int tile_column = 0; tile_column < tile_columns;
                 ++tile_column) {
                const size_t tile =
                    (size_t)tile_row * tile_columns + tile_column;
                printf("TILE_DIAG name=%s row_tile=%d column_tile=%d "
                       "mismatches=%zu zeros=%zu elements=2048\n",
                       name, tile_row, tile_column, tile_mismatches[tile],
                       tile_zeros[tile]);
            }
        }
    }

    free(tile_mismatches);
    free(tile_zeros);
    free(dense_row);
    free(dense_rank);
    free(hmx_rank);
    free(selected);
    free(query_row_major);
    free(key_row_major);
    free(packed_reference);
    return mismatches == 0 && nondegenerate ? 0 : -1;
}

static int append_unique_case(int32_t *values, int count, int32_t value) {
    for (int index = 0; index < count; ++index) {
        if (values[index] == value) return count;
    }
    values[count] = value;
    return count + 1;
}

static int run_test(
    const operator_api *api,
    hmx_i8_context *context,
    const hmx_qk_i8_operator_info *info) {
    int result = 0;
    int32_t n_cases[5];
    int n_count = 0;
    n_count = append_unique_case(n_cases, n_count, info->m);
    n_count = append_unique_case(
        n_cases, n_count, info->n < 256 ? info->n : 256);
    n_count = append_unique_case(
        n_cases, n_count, info->n < 1024 ? info->n : 1024);
    n_count = append_unique_case(
        n_cases, n_count, info->n < 2048 ? info->n : 2048);
    n_count = append_unique_case(n_cases, n_count, info->n);
    const char *quick_value = getenv("HMX_OPERATOR_TEST_QUICK");
    const int quick = quick_value && quick_value[0] != '\0' &&
        strcmp(quick_value, "0") != 0;

    for (int index = 0; index < n_count; ++index) {
        if (quick && n_cases[index] > 256) continue;
        int32_t heads = index == 0 ? 1 :
            (index == n_count - 1 ? info->heads :
             (info->heads < 3 ? info->heads : 3));
        char name[64];
        snprintf(name, sizeof(name), "direct-h%d-n%d", heads, n_cases[index]);
        const input_pattern pattern = index == 0 ? INPUT_FULL_RANGE :
            INPUT_REALISTIC;
        const int benchmark_iterations = index == n_count - 1 ? 3 : 0;
        if (verify_case(api, context, info, name, heads, n_cases[index],
                        pattern, benchmark_iterations) != 0) {
            result = -1;
        }
    }

    const int invalid_rejected =
        api->execute_hn(context, 0, info->m) != HMX_I8_OK &&
        api->execute_hn(context, info->heads + 1, info->m) != HMX_I8_OK &&
        api->execute_hn(context, 1, info->m - 1) != HMX_I8_OK &&
        api->execute_hn(context, 1, info->n + 32) != HMX_I8_OK;
    printf("INVALID_SHAPE_TEST status=%s\n",
           invalid_rejected ? "PASS" : "FAIL");
    if (!invalid_rejected) result = -1;

    printf("OPERATOR_TEST status=%s sq=%.9g sk=%.9g sy=%.9g rq=%.9g\n",
           result == 0 ? "PASS" : "FAIL", info->q_scale, info->k_scale,
           info->output_scale, info->requant_scale);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s operator.so\n", argv[0]);
        return 2;
    }
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        fprintf(stderr, "dlopen(%s): %s\n", argv[1], dlerror());
        return 1;
    }

    operator_api api = {0};
    if (load_api(library, &api) || verify_runtime_entry_hidden(library)) {
        dlclose(library);
        return 1;
    }
    if (api.api_version() != HMX_QK_I8_OPERATOR_API_VERSION) {
        fprintf(stderr, "unsupported operator ABI version %u\n", api.api_version());
        dlclose(library);
        return 1;
    }

    hmx_qk_i8_operator_info info = {0};
    int status = api.get_info(&info);
    if (status != HMX_I8_OK || validate_info(&info)) {
        fprintf(stderr, "invalid operator metadata (status=%d)\n", status);
        dlclose(library);
        return 1;
    }

    hmx_i8_context *context = NULL;
    status = api.create(&context);
    if (status != HMX_I8_OK) {
        fprintf(stderr, "hmx_i8_create: %s (%d)\n", api.status_string(status), status);
        dlclose(library);
        return 1;
    }
    int result = run_test(&api, context, &info);
    api.destroy(context);
    dlclose(library);
    return result == 0 ? 0 : 1;
}
