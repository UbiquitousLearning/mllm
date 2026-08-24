#include "hmx_i8_gemm.h"

#include <dsp_capabilities_utils.h>
#include <limits.h>
#include <math.h>
#include <remote.h>
#include <rpcmem.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hmx_int8_rpc.h"
#include "hmx_i8_dsp_timing.h"

#if defined(HMX_RPC_SKEL_FILENAME)
#define HMX_RPC_URI \
    "file:///" HMX_RPC_SKEL_FILENAME \
    "?hmx_int8_rpc_skel_handle_invoke&_modver=1.0&_idlver=7.0.0"
#else
#define HMX_RPC_URI hmx_int8_rpc_URI
#endif

#if !defined(HMX_FIXED_M) || !defined(HMX_FIXED_K) || !defined(HMX_FIXED_N) || \
    !defined(HMX_FIXED_HEADS)
#error "HMX_FIXED_HEADS/M/K/N must be defined by the build"
#endif

enum {
    HMX_M_TILE = 64,
    HMX_K_TILE = 32,
    HMX_N_TILE = 32,
    HMX_MAX_PADDED_K = 65536,
    SHARED_ALIGNMENT = 4096,
};

typedef struct shared_layout {
    int32_t padded_m;
    int32_t padded_k;
    int32_t padded_n;
    int32_t lhs_offset;
    int32_t rhs_offset;
    int32_t rhs_sums_offset;
    int32_t output_offset;
    int32_t raw_query_offset;
    int32_t raw_key_offset;
    int32_t scales_offset;
    size_t lhs_bytes;
    size_t rhs_bytes;
    size_t rhs_sums_bytes;
    size_t output_bytes;
    size_t raw_query_bytes;
    size_t raw_key_bytes;
    size_t scales_bytes;
    size_t total_bytes;
} shared_layout;

struct hmx_i8_context {
    remote_handle64 handle;
    int domain_id;
    int is_open;
    uint8_t *shared;
    int buffer_fd;
    int is_mapped;
    int arena_registered;
    int packed_scope_active;
    shared_layout layout;
};

static int checked_multiply(size_t lhs, size_t rhs, size_t *result) {
    if (lhs != 0 && rhs > SIZE_MAX / lhs) {
        return -1;
    }
    *result = lhs * rhs;
    return 0;
}

static int checked_align(size_t value, size_t alignment, size_t *result) {
    if (value > SIZE_MAX - (alignment - 1u)) {
        return -1;
    }
    *result = (value + alignment - 1u) & ~(alignment - 1u);
    return 0;
}

static int32_t round_dimension(int32_t value, int32_t tile) {
    if (value <= 0 || value > INT32_MAX - (tile - 1)) {
        return 0;
    }
    return ((value + tile - 1) / tile) * tile;
}

static int make_shared_layout(
    int32_t m,
    int32_t k,
    int32_t n,
    int32_t heads,
    size_t output_element_size,
    shared_layout *layout) {
    memset(layout, 0, sizeof(*layout));
    layout->padded_m = round_dimension(m, HMX_M_TILE);
    layout->padded_k = round_dimension(k, HMX_K_TILE);
    layout->padded_n = round_dimension(n, HMX_N_TILE);
    if (!layout->padded_m || !layout->padded_k || !layout->padded_n || heads <= 0 ||
        layout->padded_k > HMX_MAX_PADDED_K) {
        return HMX_I8_SIZE_OVERFLOW;
    }

    if (checked_multiply(
            (size_t)layout->padded_m,
            (size_t)layout->padded_k,
            &layout->lhs_bytes) ||
        checked_multiply(
            (size_t)layout->padded_k,
            (size_t)layout->padded_n,
            &layout->rhs_bytes) ||
        checked_multiply(
            (size_t)layout->padded_m,
            (size_t)layout->padded_n,
            &layout->output_bytes) ||
        checked_multiply(
            layout->output_bytes,
            output_element_size,
            &layout->output_bytes) ||
        checked_multiply(layout->lhs_bytes, (size_t)heads, &layout->lhs_bytes) ||
        checked_multiply(layout->rhs_bytes, (size_t)heads, &layout->rhs_bytes) ||
        checked_multiply((size_t)layout->padded_n, sizeof(int32_t),
                         &layout->rhs_sums_bytes) ||
        checked_multiply(layout->rhs_sums_bytes, (size_t)heads,
                         &layout->rhs_sums_bytes) ||
        checked_multiply(
            layout->output_bytes,
            (size_t)heads,
            &layout->output_bytes) ||
        checked_multiply(layout->lhs_bytes, sizeof(float),
                         &layout->raw_query_bytes) ||
        checked_multiply(layout->rhs_bytes, sizeof(uint16_t),
                         &layout->raw_key_bytes)) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    layout->scales_bytes = (size_t)hmx_i8_dsp_timing_offset(heads)
        + sizeof(hmx_i8_dsp_timing);

    size_t cursor = 0;
    if (checked_align(cursor, SHARED_ALIGNMENT, &cursor) || cursor > INT32_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    layout->lhs_offset = (int32_t)cursor;
    if (layout->lhs_bytes > SIZE_MAX - cursor) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    cursor += layout->lhs_bytes;

    if (checked_align(cursor, SHARED_ALIGNMENT, &cursor) || cursor > INT32_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    layout->rhs_offset = (int32_t)cursor;
    if (layout->rhs_bytes > SIZE_MAX - cursor) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    cursor += layout->rhs_bytes;

    if (checked_align(cursor, SHARED_ALIGNMENT, &cursor) || cursor > INT32_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    layout->rhs_sums_offset = (int32_t)cursor;
    if (layout->rhs_sums_bytes > SIZE_MAX - cursor) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    cursor += layout->rhs_sums_bytes;

    if (checked_align(cursor, SHARED_ALIGNMENT, &cursor) || cursor > INT32_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    layout->output_offset = (int32_t)cursor;
    if (layout->output_bytes > SIZE_MAX - cursor) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    cursor += layout->output_bytes;

    if (checked_align(cursor, SHARED_ALIGNMENT, &cursor) || cursor > INT32_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    layout->raw_query_offset = (int32_t)cursor;
    if (layout->raw_query_bytes > SIZE_MAX - cursor) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    cursor += layout->raw_query_bytes;

    if (checked_align(cursor, SHARED_ALIGNMENT, &cursor) || cursor > INT32_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    layout->raw_key_offset = (int32_t)cursor;
    if (layout->raw_key_bytes > SIZE_MAX - cursor) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    cursor += layout->raw_key_bytes;

    if (checked_align(cursor, SHARED_ALIGNMENT, &cursor) || cursor > INT32_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    layout->scales_offset = (int32_t)cursor;
    if (layout->scales_bytes > SIZE_MAX - cursor) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    cursor += layout->scales_bytes;

    if (checked_align(cursor, SHARED_ALIGNMENT, &layout->total_bytes) ||
        layout->total_bytes > INT_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    return HMX_I8_OK;
}

static void release_qk_shared(hmx_i8_context *context) {
    if (!context) {
        return;
    }
    if (context->arena_registered && context->is_open) {
        (void)hmx_int8_rpc_unregister_arena(
            context->handle, context->buffer_fd);
        context->arena_registered = 0;
    }
    if (context->is_mapped) {
        (void)fastrpc_munmap(
            context->domain_id,
            context->buffer_fd,
            context->shared,
            context->layout.total_bytes);
    }
    if (context->shared) {
        rpcmem_free(context->shared);
    }
    context->shared = NULL;
    context->buffer_fd = -1;
    context->is_mapped = 0;
    context->arena_registered = 0;
    memset(&context->layout, 0, sizeof(context->layout));
}

#if defined(HMX_QK_OPERATOR_BUILD)
static int reserve_qk_shared(hmx_i8_context *context) {
    if (!context || !context->is_open) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    if (context->shared && context->is_mapped) {
        return HMX_I8_OK;
    }

    shared_layout layout;
    int status = make_shared_layout(
        HMX_FIXED_M,
        HMX_FIXED_K,
        HMX_FIXED_N,
        HMX_FIXED_HEADS,
        sizeof(int8_t),
        &layout);
    if (status != HMX_I8_OK) {
        return status;
    }
    uint8_t *shared = (uint8_t *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_FLAG_UNCACHED,
        (int)layout.total_bytes);
    if (!shared) {
        return HMX_I8_OUT_OF_MEMORY;
    }
    const int buffer_fd = rpcmem_to_fd(shared);
    if (buffer_fd < 0 ||
        fastrpc_mmap(
            context->domain_id,
            buffer_fd,
            shared,
            0,
            layout.total_bytes,
            FASTRPC_MAP_FD) != 0) {
        rpcmem_free(shared);
        return HMX_I8_FASTRPC_ERROR;
    }
    if (hmx_int8_rpc_register_arena(context->handle, buffer_fd) != 0) {
        (void)fastrpc_munmap(
            context->domain_id, buffer_fd, shared, layout.total_bytes);
        rpcmem_free(shared);
        return HMX_I8_FASTRPC_ERROR;
    }
    context->shared = shared;
    context->buffer_fd = buffer_fd;
    context->is_mapped = 1;
    context->arena_registered = 1;
    context->layout = layout;
    return HMX_I8_OK;
}
#endif

uint32_t hmx_i8_api_version(void) {
    return HMX_I8_API_VERSION;
}

int32_t hmx_i8_fixed_m(void) {
    return HMX_FIXED_M;
}

int32_t hmx_i8_fixed_k(void) {
    return HMX_FIXED_K;
}

int32_t hmx_i8_fixed_n(void) {
    return HMX_FIXED_N;
}

int hmx_i8_create(hmx_i8_context **out_context) {
    if (!out_context) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    *out_context = NULL;

    hmx_i8_context *context = (hmx_i8_context *)calloc(1, sizeof(*context));
    if (!context) {
        return HMX_I8_OUT_OF_MEMORY;
    }
    context->domain_id = CDSP_DOMAIN_ID;
    context->buffer_fd = -1;

    struct remote_rpc_control_unsigned_module unsigned_module = {
        .domain = context->domain_id,
        .enable = 1,
    };
    int rpc_status = remote_session_control(
        DSPRPC_CONTROL_UNSIGNED_MODULE,
        &unsigned_module,
        sizeof(unsigned_module));
    if (rpc_status != 0) {
        free(context);
        return HMX_I8_FASTRPC_ERROR;
    }

    domain *selected_domain = get_domain(context->domain_id);
    if (!selected_domain) {
        free(context);
        return HMX_I8_FASTRPC_ERROR;
    }

    const size_t uri_bytes =
        strlen(HMX_RPC_URI) + strlen(selected_domain->uri) + 1u;
    char *uri = (char *)malloc(uri_bytes);
    if (!uri) {
        free(context);
        return HMX_I8_OUT_OF_MEMORY;
    }
    (void)snprintf(
        uri,
        uri_bytes,
        "%s%s",
        HMX_RPC_URI,
        selected_domain->uri);

    rpc_status = hmx_int8_rpc_open(uri, &context->handle);
    free(uri);
    if (rpc_status != 0) {
        free(context);
        return HMX_I8_FASTRPC_ERROR;
    }
    context->is_open = 1;

    struct remote_rpc_control_latency latency = {
        .enable = RPC_PM_QOS,
        .latency = 100,
    };
    (void)remote_handle64_control(
        context->handle,
        DSPRPC_CONTROL_LATENCY,
        &latency,
        sizeof(latency));

    rpc_status = hmx_int8_rpc_init_backend(context->handle);
    if (rpc_status != 0) {
        hmx_int8_rpc_close(context->handle);
        free(context);
        return HMX_I8_DSP_ERROR;
    }

/* Fixed-scale ARM operators may share a larger head-capacity DSP dispatcher.
 * Generic/fixed libraries still require an exact DSP build match. */
#if !defined(HMX_QK_OPERATOR_BUILD)
    int32_t dsp_m = 0;
    int32_t dsp_k = 0;
    int32_t dsp_n = 0;
    rpc_status = hmx_int8_rpc_get_fixed_shape(
        context->handle,
        &dsp_m,
        &dsp_k,
        &dsp_n);
    if (rpc_status != 0 ||
        dsp_m != HMX_FIXED_M || dsp_k != HMX_FIXED_K || dsp_n != HMX_FIXED_N) {
        hmx_int8_rpc_close(context->handle);
        free(context);
        return HMX_I8_DSP_ERROR;
    }

    int32_t dsp_heads = 0;
    rpc_status = hmx_int8_rpc_get_fixed_heads(context->handle, &dsp_heads);
    if (rpc_status != 0 || dsp_heads != HMX_FIXED_HEADS) {
        hmx_int8_rpc_close(context->handle);
        free(context);
        return HMX_I8_DSP_ERROR;
    }

#if defined(HMX_OPERATOR_Q_SCALE) && defined(HMX_OPERATOR_K_SCALE) && \
    defined(HMX_OPERATOR_OUTPUT_SCALE)
    float dsp_q_scale = 0.0f;
    float dsp_k_scale = 0.0f;
    float dsp_output_scale = 0.0f;
    rpc_status = hmx_int8_rpc_get_fixed_scales(
        context->handle,
        &dsp_q_scale,
        &dsp_k_scale,
        &dsp_output_scale);
    if (rpc_status != 0 ||
        dsp_q_scale != (float)(HMX_OPERATOR_Q_SCALE) ||
        dsp_k_scale != (float)(HMX_OPERATOR_K_SCALE) ||
        dsp_output_scale != (float)(HMX_OPERATOR_OUTPUT_SCALE)) {
        hmx_int8_rpc_close(context->handle);
        free(context);
        return HMX_I8_DSP_ERROR;
    }
#endif
#endif

#if defined(HMX_QK_OPERATOR_BUILD)
    int reserve_status = reserve_qk_shared(context);
    if (reserve_status != HMX_I8_OK) {
        hmx_int8_rpc_close(context->handle);
        free(context);
        return reserve_status;
    }
#endif

    *out_context = context;
    return HMX_I8_OK;
}

void hmx_i8_destroy(hmx_i8_context *context) {
    if (!context) {
        return;
    }
    if (context->packed_scope_active) {
        (void)hmx_int8_rpc_end_qk_i8_packed(context->handle);
        context->packed_scope_active = 0;
    }
    release_qk_shared(context);
    if (context->is_open) {
        (void)hmx_int8_rpc_close(context->handle);
    }
    free(context);
}

#if !defined(HMX_I8_FIXED_ONLY)
static int matmul_i32_impl(
    hmx_i8_context *context,
    const int8_t *lhs,
    const int8_t *rhs,
    int32_t *output,
    int32_t m,
    int32_t k,
    int32_t n,
    int fixed_shape) {
    if (!context || !context->is_open || !lhs || !rhs || !output ||
        m <= 0 || k <= 0 || n <= 0) {
        return HMX_I8_INVALID_ARGUMENT;
    }

    shared_layout layout;
    int status = make_shared_layout(m, k, n, 1, sizeof(int32_t), &layout);
    if (status != HMX_I8_OK) {
        return status;
    }

    uint8_t *shared = (uint8_t *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_FLAG_UNCACHED,
        (int)layout.total_bytes);
    int64_t *rhs_sums = (int64_t *)calloc((size_t)n, sizeof(*rhs_sums));
    if (!shared || !rhs_sums) {
        if (shared) {
            rpcmem_free(shared);
        }
        free(rhs_sums);
        return HMX_I8_OUT_OF_MEMORY;
    }

    uint8_t *padded_lhs = shared + layout.lhs_offset;
    int8_t *padded_rhs = (int8_t *)(shared + layout.rhs_offset);
    int32_t *raw_output = (int32_t *)(shared + layout.output_offset);
    memset(padded_lhs, 128, layout.lhs_bytes);
    memset(padded_rhs, 0, layout.rhs_bytes);
    memset(raw_output, 0, layout.output_bytes);

    for (int32_t row = 0; row < m; ++row) {
        uint8_t *destination =
            padded_lhs + (size_t)row * (size_t)layout.padded_k;
        const int8_t *source = lhs + (size_t)row * (size_t)k;
        for (int32_t column = 0; column < k; ++column) {
            destination[column] = (uint8_t)((int32_t)source[column] + 128);
        }
    }
    for (int32_t row = 0; row < k; ++row) {
        memcpy(
            padded_rhs + (size_t)row * (size_t)layout.padded_n,
            rhs + (size_t)row * (size_t)n,
            (size_t)n);
        for (int32_t column = 0; column < n; ++column) {
            rhs_sums[column] += rhs[(size_t)row * (size_t)n + column];
        }
    }

    int buffer_fd = rpcmem_to_fd(shared);
    if (buffer_fd < 0) {
        free(rhs_sums);
        rpcmem_free(shared);
        return HMX_I8_FASTRPC_ERROR;
    }

    int is_mapped = 0;
    int rpc_status = fastrpc_mmap(
        context->domain_id,
        buffer_fd,
        shared,
        0,
        layout.total_bytes,
        FASTRPC_MAP_FD);
    if (rpc_status == 0) {
        is_mapped = 1;
        if (fixed_shape) {
            rpc_status = hmx_int8_rpc_matmul_fixed(
                context->handle,
                buffer_fd,
                layout.lhs_offset,
                layout.rhs_offset,
                layout.output_offset);
        } else {
            rpc_status = hmx_int8_rpc_matmul(
                context->handle,
                buffer_fd,
                layout.lhs_offset,
                layout.rhs_offset,
                layout.output_offset,
                layout.padded_m,
                layout.padded_k,
                layout.padded_n);
        }
    }

    status = HMX_I8_OK;
    if (!is_mapped) {
        status = HMX_I8_FASTRPC_ERROR;
    } else if (rpc_status != 0) {
        status = HMX_I8_DSP_ERROR;
    } else {
        for (int32_t row = 0; row < m; ++row) {
            for (int32_t column = 0; column < n; ++column) {
                int64_t corrected =
                    raw_output[(size_t)row * (size_t)layout.padded_n + column] -
                    128 * rhs_sums[column];
                if (corrected < INT32_MIN || corrected > INT32_MAX) {
                    status = HMX_I8_SIZE_OVERFLOW;
                    break;
                }
                output[(size_t)row * (size_t)n + column] = (int32_t)corrected;
            }
            if (status != HMX_I8_OK) {
                break;
            }
        }
    }

    if (is_mapped) {
        (void)fastrpc_munmap(
            context->domain_id,
            buffer_fd,
            shared,
            layout.total_bytes);
    }
    free(rhs_sums);
    rpcmem_free(shared);
    return status;
}

int hmx_i8_matmul_i32(
    hmx_i8_context *context,
    const int8_t *lhs,
    const int8_t *rhs,
    int32_t *output,
    int32_t m,
    int32_t k,
    int32_t n) {
    return matmul_i32_impl(context, lhs, rhs, output, m, k, n, 0);
}

int hmx_i8_matmul_i32_fixed(
    hmx_i8_context *context,
    const int8_t *lhs,
    const int8_t *rhs,
    int32_t *output) {
    return matmul_i32_impl(
        context,
        lhs,
        rhs,
        output,
        HMX_FIXED_M,
        HMX_FIXED_K,
        HMX_FIXED_N,
        1);
}
#endif

int hmx_i8_matmul_i8_fixed(
    hmx_i8_context *context,
    const int8_t *lhs,
    const int8_t *rhs,
    int8_t *output,
    float output_scale) {
    if (!context || !context->is_open || !lhs || !rhs || !output ||
        !isfinite(output_scale) || output_scale < 0.0f) {
        return HMX_I8_INVALID_ARGUMENT;
    }

    shared_layout layout;
    int status = make_shared_layout(
        HMX_FIXED_M,
        HMX_FIXED_K,
        HMX_FIXED_N,
        1,
        sizeof(int8_t),
        &layout);
    if (status != HMX_I8_OK) {
        return status;
    }

    uint8_t *shared = (uint8_t *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_FLAG_UNCACHED,
        (int)layout.total_bytes);
    if (!shared) {
        return HMX_I8_OUT_OF_MEMORY;
    }

    uint8_t *padded_lhs = shared + layout.lhs_offset;
    int8_t *padded_rhs = (int8_t *)(shared + layout.rhs_offset);
    int8_t *padded_output = (int8_t *)(shared + layout.output_offset);
    memset(padded_lhs, 128, layout.lhs_bytes);
    memset(padded_rhs, 0, layout.rhs_bytes);
    memset(padded_output, 0, layout.output_bytes);

    for (int32_t row = 0; row < HMX_FIXED_M; ++row) {
        uint8_t *destination =
            padded_lhs + (size_t)row * (size_t)layout.padded_k;
        const int8_t *source = lhs + (size_t)row * (size_t)HMX_FIXED_K;
        for (int32_t column = 0; column < HMX_FIXED_K; ++column) {
            destination[column] = (uint8_t)((int32_t)source[column] + 128);
        }
    }
    for (int32_t row = 0; row < HMX_FIXED_K; ++row) {
        memcpy(
            padded_rhs + (size_t)row * (size_t)layout.padded_n,
            rhs + (size_t)row * (size_t)HMX_FIXED_N,
            (size_t)HMX_FIXED_N);
    }

    int buffer_fd = rpcmem_to_fd(shared);
    if (buffer_fd < 0) {
        rpcmem_free(shared);
        return HMX_I8_FASTRPC_ERROR;
    }

    int is_mapped = 0;
    int rpc_status = fastrpc_mmap(
        context->domain_id,
        buffer_fd,
        shared,
        0,
        layout.total_bytes,
        FASTRPC_MAP_FD);
    if (rpc_status == 0) {
        is_mapped = 1;
        rpc_status = hmx_int8_rpc_matmul_i8_fixed(
            context->handle,
            buffer_fd,
            layout.lhs_offset,
            layout.rhs_offset,
            layout.output_offset,
            output_scale);
    }

    if (!is_mapped) {
        status = HMX_I8_FASTRPC_ERROR;
    } else if (rpc_status != 0) {
        status = HMX_I8_DSP_ERROR;
    } else {
        for (int32_t row = 0; row < HMX_FIXED_M; ++row) {
            memcpy(
                output + (size_t)row * (size_t)HMX_FIXED_N,
                padded_output + (size_t)row * (size_t)layout.padded_n,
                (size_t)HMX_FIXED_N);
        }
        status = HMX_I8_OK;
    }

    if (is_mapped) {
        (void)fastrpc_munmap(
            context->domain_id,
            buffer_fd,
            shared,
            layout.total_bytes);
    }
    rpcmem_free(shared);
    return status;
}

#if defined(HMX_QK_OPERATOR_BUILD)
int8_t *hmx_i8_qk_query_data(hmx_i8_context *context) {
    return context && context->shared
        ? (int8_t *)(context->shared + context->layout.lhs_offset)
        : NULL;
}

int8_t *hmx_i8_qk_key_data(hmx_i8_context *context) {
    return context && context->shared
        ? (int8_t *)(context->shared + context->layout.rhs_offset)
        : NULL;
}

int32_t *hmx_i8_qk_key_sums_data(hmx_i8_context *context) {
    return context && context->shared
        ? (int32_t *)(context->shared + context->layout.rhs_sums_offset)
        : NULL;
}

const int8_t *hmx_i8_qk_scores_data(hmx_i8_context *context) {
    return context && context->shared
        ? (const int8_t *)(context->shared + context->layout.output_offset)
        : NULL;
}

const int32_t *hmx_i8_qk_topk_indices_data(hmx_i8_context *context) {
    return context && context->shared
        ? (const int32_t *)(context->shared + context->layout.output_offset)
        : NULL;
}

float *hmx_i8_qk_raw_query_data(hmx_i8_context *context) {
    return context && context->shared
        ? (float *)(context->shared + context->layout.raw_query_offset)
        : NULL;
}

uint16_t *hmx_i8_qk_raw_key_data(hmx_i8_context *context) {
    return context && context->shared
        ? (uint16_t *)(context->shared + context->layout.raw_key_offset)
        : NULL;
}

int hmx_i8_profile_qk_raw_scales(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t profile_query, int32_t profile_key, float *q_scales,
    float *k_scales) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || !q_scales || !k_scales || heads <= 0 ||
        heads > HMX_FIXED_HEADS || query_rows <= 0 ||
        query_rows > HMX_FIXED_M || n <= 0 || n > HMX_FIXED_N_PADDED ||
        (n % HMX_N_TILE) != 0 || (!profile_query && !profile_key)) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    const int rpc_status = hmx_int8_rpc_profile_qk_raw_scales(
        context->handle, context->buffer_fd,
        context->layout.raw_query_offset, context->layout.raw_key_offset,
        context->layout.scales_offset, heads, query_rows, n,
        profile_query, profile_key);
    if (rpc_status != 0) return HMX_I8_DSP_ERROR;
    const float *scales = (const float *)(context->shared
                                          + context->layout.scales_offset);
    memcpy(q_scales, scales, (size_t)heads * sizeof(float));
    memcpy(k_scales, scales + heads, (size_t)heads * sizeof(float));
    return HMX_I8_OK;
}

int hmx_i8_profile_qk_raw_scales_incremental(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, int32_t profile_query,
    const float *previous_k_scales, float *q_scales, float *k_scales) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || !previous_k_scales || !q_scales ||
        !k_scales || heads <= 0 || heads > HMX_FIXED_HEADS ||
        query_rows <= 0 || query_rows > HMX_FIXED_M || n <= 0 ||
        n > HMX_FIXED_N_PADDED || (n % HMX_N_TILE) != 0 ||
        key_begin < 0 || key_begin >= n ||
        key_begin > INT32_MAX - 2) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    float *scales = (float *)(context->shared
                              + context->layout.scales_offset);
    for (int32_t head = 0; head < heads; ++head) {
        if (!isfinite(previous_k_scales[head]) ||
            previous_k_scales[head] < 0.0f) {
            return HMX_I8_INVALID_ARGUMENT;
        }
        scales[heads + head] = previous_k_scales[head];
    }
    const int32_t incremental_key = key_begin + 2;
    const int rpc_status = hmx_int8_rpc_profile_qk_raw_scales(
        context->handle, context->buffer_fd,
        context->layout.raw_query_offset, context->layout.raw_key_offset,
        context->layout.scales_offset, heads, query_rows, n,
        profile_query != 0, incremental_key);
    if (rpc_status != 0) return HMX_I8_DSP_ERROR;
    memcpy(q_scales, scales, (size_t)heads * sizeof(float));
    memcpy(k_scales, scales + heads, (size_t)heads * sizeof(float));
    return HMX_I8_OK;
}

int hmx_i8_last_dsp_timing(
    hmx_i8_context *context, hmx_i8_dsp_timing *timing) {
    if (!context || !context->shared || !timing) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    const uint8_t *base = context->shared + context->layout.scales_offset
        + hmx_i8_dsp_timing_offset(HMX_FIXED_HEADS);
    memcpy(timing, base, sizeof(*timing));
    if (timing->struct_size != sizeof(*timing)
        || timing->version != HMX_I8_DSP_TIMING_VERSION) {
        return HMX_I8_DSP_ERROR;
    }
    return HMX_I8_OK;
}

int hmx_i8_prepare_qk_i8_raw_key_scaled(
    hmx_i8_context *context, int32_t heads, int32_t n, int32_t key_begin,
    float k_scale) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || heads <= 0 || heads > HMX_FIXED_HEADS ||
        n <= 0 || n > HMX_FIXED_N_PADDED || (n % HMX_N_TILE) != 0 ||
        key_begin < 0 || key_begin > n || !isfinite(k_scale) ||
        k_scale <= 0.0f) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    const int rpc_status = hmx_int8_rpc_prepare_qk_i8_raw_key(
        context->handle, context->buffer_fd,
        context->layout.raw_key_offset, context->layout.rhs_offset,
        context->layout.rhs_sums_offset, heads, n, key_begin,
        1.0f / k_scale);
    return rpc_status == 0 ? HMX_I8_OK : HMX_I8_DSP_ERROR;
}

int hmx_i8_prepare_qk_i8_raw_key(
    hmx_i8_context *context, int32_t heads, int32_t n, int32_t key_begin) {
    return hmx_i8_prepare_qk_i8_raw_key_scaled(
        context, heads, n, key_begin, (float)(HMX_OPERATOR_K_SCALE));
}

int hmx_i8_execute_qk_i8_raw_dynamic(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    float q_scale, float requant_scale) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || heads <= 0 || heads > HMX_FIXED_HEADS ||
        query_rows <= 0 || query_rows > HMX_FIXED_M || n <= 0 ||
        n > HMX_FIXED_N_PADDED || (n % HMX_N_TILE) != 0 ||
        !isfinite(q_scale) || q_scale <= 0.0f ||
        !isfinite(requant_scale) || requant_scale <= 0.0f) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    const int rpc_status = hmx_int8_rpc_execute_qk_i8_raw_query(
        context->handle, context->buffer_fd,
        context->layout.raw_query_offset, context->layout.lhs_offset,
        context->layout.rhs_offset, context->layout.rhs_sums_offset,
        context->layout.output_offset, heads, query_rows, n,
        1.0f / q_scale,
        requant_scale);
    return rpc_status == 0 ? HMX_I8_OK : HMX_I8_DSP_ERROR;
}

int hmx_i8_execute_qk_i32_topk_raw(
    hmx_i8_context *context, int32_t query_rows, int32_t key_len, int32_t n,
    int32_t causal_prefix_tokens, float q_scale,
    const int32_t *row_offsets, int32_t row_offsets_count) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || query_rows <= 0 ||
        query_rows > HMX_FIXED_M || key_len <= 0 || key_len > n ||
        n <= 0 || n > HMX_FIXED_N_PADDED || (n % HMX_N_TILE) != 0 ||
        causal_prefix_tokens < 0 || !isfinite(q_scale) || q_scale <= 0.0f ||
        !row_offsets || row_offsets_count != query_rows + 1) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    const size_t score_bytes = (size_t)context->layout.padded_m
        * (size_t)n * sizeof(int32_t);
    const size_t offsets_bytes = (size_t)row_offsets_count * sizeof(int32_t);
    if (score_bytes > context->layout.output_bytes ||
        offsets_bytes > context->layout.output_bytes - score_bytes ||
        (size_t)context->layout.output_offset + score_bytes > INT32_MAX) {
        return HMX_I8_SIZE_OVERFLOW;
    }
    const int32_t row_offsets_offset = context->layout.output_offset
        + (int32_t)score_bytes;
    memcpy(context->shared + row_offsets_offset, row_offsets, offsets_bytes);
    const int rpc_status = hmx_int8_rpc_execute_qk_i32_topk_raw_query(
        context->handle, context->buffer_fd,
        context->layout.raw_query_offset, context->layout.lhs_offset,
        context->layout.rhs_offset, context->layout.rhs_sums_offset,
        context->layout.output_offset, row_offsets_offset, query_rows,
        key_len, n, causal_prefix_tokens, 1.0f / q_scale);
    return rpc_status == 0 ? HMX_I8_OK : HMX_I8_DSP_ERROR;
}

int hmx_i8_execute_qk_i8_raw_scaled(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    float requant_scale) {
    return hmx_i8_execute_qk_i8_raw_dynamic(
        context, heads, query_rows, n, (float)(HMX_OPERATOR_Q_SCALE),
        requant_scale);
}

int hmx_i8_execute_qk_i8_raw(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n) {
    return hmx_i8_execute_qk_i8_raw_scaled(
        context, heads, query_rows, n,
        (float)(HMX_OPERATOR_Q_SCALE) * (float)(HMX_OPERATOR_K_SCALE) /
            (float)(HMX_OPERATOR_OUTPUT_SCALE));
}

int hmx_i8_prepare_execute_qk_i8_raw(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || !context->arena_registered || heads <= 0 ||
        heads > HMX_FIXED_HEADS || query_rows <= 0 ||
        query_rows > HMX_FIXED_M || n <= 0 || n > HMX_FIXED_N_PADDED ||
        (n % HMX_N_TILE) != 0 || key_begin < 0 || key_begin > n) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    const int rpc_status = hmx_int8_rpc_prepare_execute_qk_i8_raw(
        context->handle, context->buffer_fd,
        context->layout.raw_query_offset, context->layout.raw_key_offset,
        context->layout.lhs_offset, context->layout.rhs_offset,
        context->layout.rhs_sums_offset, context->layout.output_offset,
        heads, query_rows, n, key_begin,
        1.0f / (float)(HMX_OPERATOR_Q_SCALE),
        1.0f / (float)(HMX_OPERATOR_K_SCALE),
        (float)(HMX_OPERATOR_Q_SCALE) * (float)(HMX_OPERATOR_K_SCALE) /
            (float)(HMX_OPERATOR_OUTPUT_SCALE));
    return rpc_status == 0 ? HMX_I8_OK : HMX_I8_DSP_ERROR;
}

int hmx_i8_prepare_execute_qk_i8_raw_per_head(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, const float *q_scales, const float *k_scales,
    float output_scale) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || !context->arena_registered || !q_scales ||
        !k_scales || heads <= 0 || heads > HMX_FIXED_HEADS ||
        query_rows <= 0 || query_rows > HMX_FIXED_M || n <= 0 ||
        n > HMX_FIXED_N_PADDED || (n % HMX_N_TILE) != 0 || key_begin < 0 ||
        key_begin > n || !isfinite(output_scale) || output_scale <= 0.0f) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    float *scales = (float *)(context->shared + context->layout.scales_offset);
    float *inverse_q = scales;
    float *inverse_k = scales + heads;
    float *requant = scales + heads * 2;
    for (int32_t head = 0; head < heads; ++head) {
        if (!isfinite(q_scales[head]) || q_scales[head] <= 0.0f ||
            !isfinite(k_scales[head]) || k_scales[head] <= 0.0f) {
            return HMX_I8_INVALID_ARGUMENT;
        }
        inverse_q[head] = 1.0f / q_scales[head];
        inverse_k[head] = 1.0f / k_scales[head];
        requant[head] = q_scales[head] * k_scales[head] / output_scale;
    }
    const int rpc_status = hmx_int8_rpc_prepare_execute_qk_i8_raw_per_head(
        context->handle, context->buffer_fd,
        context->layout.raw_query_offset, context->layout.raw_key_offset,
        context->layout.lhs_offset, context->layout.rhs_offset,
        context->layout.rhs_sums_offset, context->layout.output_offset,
        context->layout.scales_offset, heads, query_rows, n, key_begin);
    return rpc_status == 0 ? HMX_I8_OK : HMX_I8_DSP_ERROR;
}

int hmx_i8_prepare_execute_qk_i8_raw_per_head_requant(
    hmx_i8_context *context, int32_t heads, int32_t query_rows, int32_t n,
    int32_t key_begin, const float *q_scales, const float *k_scales,
    const float *requant_scales) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || !context->arena_registered || !q_scales ||
        !k_scales || !requant_scales || heads <= 0 ||
        heads > HMX_FIXED_HEADS || query_rows <= 0 ||
        query_rows > HMX_FIXED_M || n <= 0 || n > HMX_FIXED_N_PADDED ||
        (n % HMX_N_TILE) != 0 || key_begin < 0 || key_begin > n) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    float *scales = (float *)(context->shared + context->layout.scales_offset);
    float *inverse_q = scales;
    float *inverse_k = scales + heads;
    float *requant = scales + heads * 2;
    for (int32_t head = 0; head < heads; ++head) {
        if (!isfinite(q_scales[head]) || q_scales[head] <= 0.0f ||
            !isfinite(k_scales[head]) || k_scales[head] <= 0.0f ||
            !isfinite(requant_scales[head]) ||
            requant_scales[head] < 0.0f) {
            return HMX_I8_INVALID_ARGUMENT;
        }
        inverse_q[head] = 1.0f / q_scales[head];
        inverse_k[head] = 1.0f / k_scales[head];
        requant[head] = requant_scales[head];
    }
    const int rpc_status = hmx_int8_rpc_prepare_execute_qk_i8_raw_per_head(
        context->handle, context->buffer_fd,
        context->layout.raw_query_offset, context->layout.raw_key_offset,
        context->layout.lhs_offset, context->layout.rhs_offset,
        context->layout.rhs_sums_offset, context->layout.output_offset,
        context->layout.scales_offset, heads, query_rows, n, key_begin);
    return rpc_status == 0 ? HMX_I8_OK : HMX_I8_DSP_ERROR;
}

int hmx_i8_execute_qk_i8_direct(
    hmx_i8_context *context, int32_t heads, int32_t n) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || heads <= 0 || heads > HMX_FIXED_HEADS ||
        n <= 0 || n > HMX_FIXED_N_PADDED || (n % HMX_N_TILE) != 0) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    const int rpc_status = hmx_int8_rpc_execute_qk_i8_packed(
        context->handle,
        context->buffer_fd,
        context->layout.lhs_offset,
        context->layout.rhs_offset,
        context->layout.rhs_sums_offset,
        context->layout.output_offset,
        heads,
        n,
        (float)(HMX_OPERATOR_Q_SCALE) * (float)(HMX_OPERATOR_K_SCALE) /
            (float)(HMX_OPERATOR_OUTPUT_SCALE));
    return rpc_status == 0 ? HMX_I8_OK : HMX_I8_DSP_ERROR;
}

volatile int32_t *hmx_i8_qk_cpu_pack_ready_data(hmx_i8_context *context) {
    return context && context->shared
        ? (volatile int32_t *)(context->shared
                               + context->layout.raw_query_offset)
        : NULL;
}

int hmx_i8_execute_qk_i8_cpu_packed_per_head(
    hmx_i8_context *context, int32_t heads, int32_t n,
    const float *requant_scales) {
    if (!context || !context->is_open || !context->shared ||
        !context->is_mapped || !context->arena_registered ||
        !requant_scales || heads <= 0 || heads > HMX_FIXED_HEADS ||
        n <= 0 || n > HMX_FIXED_N_PADDED || (n % HMX_N_TILE) != 0) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    float *scales = (float *)(context->shared + context->layout.scales_offset);
    float *requant = scales + heads * 2;
    for (int32_t head = 0; head < heads; ++head) {
        if (!isfinite(requant_scales[head]) || requant_scales[head] < 0.0f) {
            return HMX_I8_INVALID_ARGUMENT;
        }
        requant[head] = requant_scales[head];
    }
    const int rpc_status = hmx_int8_rpc_execute_qk_i8_cpu_packed_per_head(
        context->handle, context->buffer_fd,
        context->layout.lhs_offset, context->layout.rhs_offset,
        context->layout.rhs_sums_offset, context->layout.output_offset,
        context->layout.scales_offset, context->layout.raw_query_offset,
        heads, n);
    return rpc_status == 0 ? HMX_I8_OK : HMX_I8_DSP_ERROR;
}

int hmx_i8_begin_qk_i8_packed(hmx_i8_context *context, int32_t n) {
    if (!context || !context->is_open || n <= 0 ||
        n > HMX_FIXED_N_PADDED || (n % HMX_N_TILE) != 0 ||
        context->packed_scope_active) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    if (hmx_int8_rpc_begin_qk_i8_packed(context->handle, n) != 0) {
        return HMX_I8_DSP_ERROR;
    }
    context->packed_scope_active = 1;
    return HMX_I8_OK;
}

int hmx_i8_end_qk_i8_packed(hmx_i8_context *context) {
    if (!context || !context->is_open || !context->packed_scope_active) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    if (hmx_int8_rpc_end_qk_i8_packed(context->handle) != 0) {
        return HMX_I8_DSP_ERROR;
    }
    context->packed_scope_active = 0;
    return HMX_I8_OK;
}

int hmx_i8_matmul_qk_i8_direct(
    hmx_i8_context *context,
    const int8_t *query,
    const int8_t *key,
    int8_t *score,
    int32_t heads,
    int32_t n) {
    if (!query || !key || !score || heads <= 0 || n <= 0 ||
        (n % HMX_N_TILE) != 0) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    int8_t *shared_query = hmx_i8_qk_query_data(context);
    int8_t *shared_key = hmx_i8_qk_key_data(context);
    int32_t *shared_sums = hmx_i8_qk_key_sums_data(context);
    if (!shared_query || !shared_key || !shared_sums) {
        return HMX_I8_INVALID_ARGUMENT;
    }
    const size_t query_per_head =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_K_PADDED;
    const size_t key_per_head = (size_t)n * HMX_FIXED_K_PADDED;
    const size_t query_bytes = (size_t)heads * query_per_head;
    const size_t key_bytes = (size_t)heads * key_per_head;
    const size_t score_bytes =
        (size_t)heads * HMX_FIXED_M_PADDED * (size_t)n;
    memset(shared_query, 0, query_bytes);
    memset(shared_key, 0, key_bytes);
    memset(shared_sums, 0, (size_t)heads * (size_t)n * sizeof(int32_t));
    const int32_t k_tiles = HMX_FIXED_K_PADDED / HMX_K_TILE;
    for (int32_t head = 0; head < heads; ++head) {
        uint8_t *packed_query = (uint8_t *)shared_query
            + (size_t)head * query_per_head;
        int8_t *packed_key = shared_key + (size_t)head * key_per_head;
        int32_t *key_sums = shared_sums + (size_t)head * (size_t)n;
        for (int32_t row = 0; row < HMX_FIXED_M_PADDED; ++row) {
            for (int32_t inner = 0; inner < HMX_FIXED_K_PADDED; ++inner) {
                int8_t value = 0;
                if (row < HMX_FIXED_M && inner < HMX_FIXED_K) {
                    value = query[((size_t)head * HMX_FIXED_M + row)
                                  * HMX_FIXED_K + inner];
                }
                const size_t packed_index =
                    ((size_t)(row / HMX_M_TILE) * k_tiles
                     + (size_t)(inner / HMX_K_TILE))
                        * HMX_M_TILE * HMX_K_TILE
                    + (size_t)(row % HMX_M_TILE) * HMX_K_TILE
                    + (size_t)(inner % HMX_K_TILE);
                packed_query[packed_index] = (uint8_t)value ^ UINT8_C(0x80);
            }
        }
        for (int32_t token = 0; token < n; ++token) {
            int32_t sum = 0;
            for (int32_t inner = 0; inner < HMX_FIXED_K_PADDED; ++inner) {
                int8_t value = 0;
                if (inner < HMX_FIXED_K && token < HMX_FIXED_N) {
                    value = key[((size_t)head * HMX_FIXED_N + token)
                                * HMX_FIXED_K + inner];
                }
                sum += value;
                const size_t packed_index =
                    ((size_t)(token / HMX_N_TILE) * k_tiles
                     + (size_t)(inner / HMX_K_TILE))
                        * HMX_K_TILE * HMX_N_TILE
                    + (size_t)((inner % HMX_K_TILE) / 4) * 128u
                    + (size_t)(token % HMX_N_TILE) * 4u
                    + (size_t)(inner % 4);
                packed_key[packed_index] = value;
            }
            key_sums[token] = sum;
        }
    }
    int status = hmx_i8_execute_qk_i8_direct(context, heads, n);
    if (status == HMX_I8_OK) {
        memcpy(score, hmx_i8_qk_scores_data(context), score_bytes);
    }
    return status;
}

/* Internal implementation used by the fixed-scale QK operator wrapper. Q and
 * K use BHSD order with B=1: [H,M,K] and [H,N,K], respectively. */
int hmx_i8_matmul_qk_i8_fixed(
    hmx_i8_context *context,
    const int8_t *query,
    const int8_t *key,
    int8_t *score) {
#if HMX_FIXED_M == HMX_FIXED_M_PADDED && \
    HMX_FIXED_K == HMX_FIXED_K_PADDED && \
    HMX_FIXED_N == HMX_FIXED_N_PADDED
    return hmx_i8_matmul_qk_i8_direct(
        context,
        query,
        key,
        score,
        HMX_FIXED_HEADS,
        HMX_FIXED_N_PADDED);
#else
    if (!context || !context->is_open || !query || !key || !score) {
        return HMX_I8_INVALID_ARGUMENT;
    }

    shared_layout layout;
    int status = make_shared_layout(
        HMX_FIXED_M,
        HMX_FIXED_K,
        HMX_FIXED_N,
        HMX_FIXED_HEADS,
        sizeof(int8_t),
        &layout);
    if (status != HMX_I8_OK) {
        return status;
    }

    uint8_t *shared = (uint8_t *)rpcmem_alloc(
        RPCMEM_HEAP_ID_SYSTEM,
        RPCMEM_FLAG_UNCACHED,
        (int)layout.total_bytes);
    if (!shared) {
        return HMX_I8_OUT_OF_MEMORY;
    }

    uint8_t *padded_query = shared + layout.lhs_offset;
    int8_t *padded_key = (int8_t *)(shared + layout.rhs_offset);
    int8_t *padded_score = (int8_t *)(shared + layout.output_offset);
    memset(padded_query, 128, layout.lhs_bytes);
    memset(padded_key, 0, layout.rhs_bytes);
    memset(padded_score, 0, layout.output_bytes);

    const size_t query_stride = (size_t)HMX_FIXED_M * HMX_FIXED_K;
    const size_t key_stride = (size_t)HMX_FIXED_N * HMX_FIXED_K;
    const size_t score_stride = (size_t)HMX_FIXED_M * HMX_FIXED_N;
    const size_t padded_query_stride =
        (size_t)layout.padded_m * layout.padded_k;
    const size_t padded_key_stride =
        (size_t)layout.padded_n * layout.padded_k;

    for (int32_t head = 0; head < HMX_FIXED_HEADS; ++head) {
        uint8_t *head_query =
            padded_query + (size_t)head * padded_query_stride;
        const int8_t *source_query = query + (size_t)head * query_stride;
        for (int32_t row = 0; row < HMX_FIXED_M; ++row) {
            uint8_t *destination =
                head_query + (size_t)row * layout.padded_k;
            const int8_t *source =
                source_query + (size_t)row * HMX_FIXED_K;
            for (int32_t inner = 0; inner < HMX_FIXED_K; ++inner) {
                destination[inner] = (uint8_t)((int32_t)source[inner] + 128);
            }
        }

        int8_t *head_key = padded_key + (size_t)head * padded_key_stride;
        const int8_t *source_key = key + (size_t)head * key_stride;
        for (int32_t token = 0; token < HMX_FIXED_N; ++token) {
            memcpy(
                head_key + (size_t)token * layout.padded_k,
                source_key + (size_t)token * HMX_FIXED_K,
                (size_t)HMX_FIXED_K);
        }
    }

    int buffer_fd = rpcmem_to_fd(shared);
    if (buffer_fd < 0) {
        rpcmem_free(shared);
        return HMX_I8_FASTRPC_ERROR;
    }

    int is_mapped = 0;
    int rpc_status = fastrpc_mmap(
        context->domain_id,
        buffer_fd,
        shared,
        0,
        layout.total_bytes,
        FASTRPC_MAP_FD);
    if (rpc_status == 0) {
        is_mapped = 1;
        rpc_status = hmx_int8_rpc_matmul_qk_i8_fixed(
            context->handle,
            buffer_fd,
            layout.lhs_offset,
            layout.rhs_offset,
            layout.output_offset);
    }

    if (!is_mapped) {
        status = HMX_I8_FASTRPC_ERROR;
    } else if (rpc_status != 0) {
        status = HMX_I8_DSP_ERROR;
    } else {
        const size_t padded_score_stride =
            (size_t)layout.padded_m * layout.padded_n;
        for (int32_t head = 0; head < HMX_FIXED_HEADS; ++head) {
            const int8_t *head_padded_score =
                padded_score + (size_t)head * padded_score_stride;
            int8_t *head_score = score + (size_t)head * score_stride;
            for (int32_t row = 0; row < HMX_FIXED_M; ++row) {
                memcpy(
                    head_score + (size_t)row * HMX_FIXED_N,
                    head_padded_score + (size_t)row * layout.padded_n,
                    (size_t)HMX_FIXED_N);
            }
        }
        status = HMX_I8_OK;
    }

    if (is_mapped) {
        (void)fastrpc_munmap(
            context->domain_id,
            buffer_fd,
            shared,
            layout.total_bytes);
    }
    rpcmem_free(shared);
    return status;
#endif
}
#endif

#if !defined(HMX_I8_FIXED_ONLY)
static int quantization_scale(
    const float *values,
    size_t count,
    float *scale) {
    float maximum = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(values[index])) {
            return HMX_I8_INVALID_ARGUMENT;
        }
        float magnitude = fabsf(values[index]);
        if (magnitude > maximum) {
            maximum = magnitude;
        }
    }
    *scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
    return HMX_I8_OK;
}

static void quantize_symmetric(
    int8_t *output,
    const float *input,
    size_t count,
    float scale) {
    for (size_t index = 0; index < count; ++index) {
        long quantized = lrintf(input[index] / scale);
        if (quantized < -127) {
            quantized = -127;
        } else if (quantized > 127) {
            quantized = 127;
        }
        output[index] = (int8_t)quantized;
    }
}

static int matmul_f32_impl(
    hmx_i8_context *context,
    const float *lhs,
    const float *rhs,
    float *output,
    int32_t m,
    int32_t k,
    int32_t n,
    float *lhs_scale,
    float *rhs_scale,
    int fixed_shape) {
    if (!context || !lhs || !rhs || !output || m <= 0 || k <= 0 || n <= 0) {
        return HMX_I8_INVALID_ARGUMENT;
    }

    size_t lhs_count;
    size_t rhs_count;
    size_t output_count;
    size_t output_bytes;
    if (checked_multiply((size_t)m, (size_t)k, &lhs_count) ||
        checked_multiply((size_t)k, (size_t)n, &rhs_count) ||
        checked_multiply((size_t)m, (size_t)n, &output_count) ||
        checked_multiply(output_count, sizeof(int32_t), &output_bytes)) {
        return HMX_I8_SIZE_OVERFLOW;
    }

    float lhs_quant_scale;
    float rhs_quant_scale;
    int status = quantization_scale(lhs, lhs_count, &lhs_quant_scale);
    if (status == HMX_I8_OK) {
        status = quantization_scale(rhs, rhs_count, &rhs_quant_scale);
    }
    if (status != HMX_I8_OK) {
        return status;
    }

    int8_t *quantized_lhs = (int8_t *)malloc(lhs_count);
    int8_t *quantized_rhs = (int8_t *)malloc(rhs_count);
    int32_t *quantized_output = (int32_t *)malloc(output_bytes);
    if (!quantized_lhs || !quantized_rhs || !quantized_output) {
        free(quantized_lhs);
        free(quantized_rhs);
        free(quantized_output);
        return HMX_I8_OUT_OF_MEMORY;
    }

    quantize_symmetric(quantized_lhs, lhs, lhs_count, lhs_quant_scale);
    quantize_symmetric(quantized_rhs, rhs, rhs_count, rhs_quant_scale);
    if (fixed_shape) {
        status = hmx_i8_matmul_i32_fixed(
            context,
            quantized_lhs,
            quantized_rhs,
            quantized_output);
    } else {
        status = hmx_i8_matmul_i32(
            context,
            quantized_lhs,
            quantized_rhs,
            quantized_output,
            m,
            k,
            n);
    }
    if (status == HMX_I8_OK) {
        const float output_scale = lhs_quant_scale * rhs_quant_scale;
        for (size_t index = 0; index < output_count; ++index) {
            output[index] = (float)quantized_output[index] * output_scale;
        }
        if (lhs_scale) {
            *lhs_scale = lhs_quant_scale;
        }
        if (rhs_scale) {
            *rhs_scale = rhs_quant_scale;
        }
    }

    free(quantized_lhs);
    free(quantized_rhs);
    free(quantized_output);
    return status;
}

int hmx_i8_matmul_f32(
    hmx_i8_context *context,
    const float *lhs,
    const float *rhs,
    float *output,
    int32_t m,
    int32_t k,
    int32_t n,
    float *lhs_scale,
    float *rhs_scale) {
    return matmul_f32_impl(
        context,
        lhs,
        rhs,
        output,
        m,
        k,
        n,
        lhs_scale,
        rhs_scale,
        0);
}

int hmx_i8_matmul_f32_fixed(
    hmx_i8_context *context,
    const float *lhs,
    const float *rhs,
    float *output,
    float *lhs_scale,
    float *rhs_scale) {
    return matmul_f32_impl(
        context,
        lhs,
        rhs,
        output,
        HMX_FIXED_M,
        HMX_FIXED_K,
        HMX_FIXED_N,
        lhs_scale,
        rhs_scale,
        1);
}
#endif

const char *hmx_i8_status_string(int status) {
    switch (status) {
        case HMX_I8_OK:
            return "success";
        case HMX_I8_INVALID_ARGUMENT:
            return "invalid argument";
        case HMX_I8_OUT_OF_MEMORY:
            return "out of memory";
        case HMX_I8_FASTRPC_ERROR:
            return "FastRPC error";
        case HMX_I8_DSP_ERROR:
            return "DSP/HMX error";
        case HMX_I8_SIZE_OVERFLOW:
            return "dimension, allocation, or accumulator overflow";
        default:
            return "unknown error";
    }
}
