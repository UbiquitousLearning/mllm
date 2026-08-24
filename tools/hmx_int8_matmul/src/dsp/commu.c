#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <HAP_mem.h>
#include <HAP_perf.h>
#include <qurt.h>
#include <qurt_memory.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>

#include "dsp/hmx_i8_kernel.h"
#include "dsp/hvx_worker_pool.h"
#include "dsp/runtime.h"
#include "hmx_i8_dsp_timing.h"
#include "hmx_int8_rpc.h"

#ifndef HMX_HVX_EPILOGUE_LANES
#define HMX_HVX_EPILOGUE_LANES 1
#endif

static int s_dummy_handle;
static hmx_runtime_session s_packed_session;
static int s_packed_session_active;
static int s_open_handles;

enum { HMX_MAX_REGISTERED_ARENAS = 32 };

typedef struct hmx_registered_arena {
    int32 buffer_fd;
    uint8_t *buffer;
    int32 users;
} hmx_registered_arena;

static hmx_registered_arena s_registered_arenas[HMX_MAX_REGISTERED_ARENAS];

static void hmx_record_worker_error(volatile int *status, int error) {
    if (error != 0) {
        (void)__sync_bool_compare_and_swap(status, 0, error);
    }
}

typedef struct hmx_scale_job {
    const float *raw_query;
    const uint16_t *raw_key;
    float *scales;
    size_t query_stride;
    size_t key_stride;
    int heads;
    int query_rows;
    int n;
    int key_begin;
    int profile_query;
    int profile_key;
    int incremental_key;
    volatile int status;
} hmx_scale_job;

static void hmx_scale_worker(void *opaque, int lane, int lanes) {
    hmx_scale_job *job = (hmx_scale_job *)opaque;
    for (int head = lane; head < job->heads; head += lanes) {
        if (job->profile_query) {
            const float *query = (const float *)((const uint8_t *)job->raw_query
                + (size_t)head * job->query_stride);
            hmx_record_worker_error(
                &job->status,
                hmx_i8_hvx_scale_f32(
                    query, job->query_rows, HMX_FIXED_K_PADDED,
                    HMX_FIXED_K_PADDED, job->scales + head));
        }
        if (job->profile_key) {
            const uint16_t *key = (const uint16_t *)((const uint8_t *)job->raw_key
                + (size_t)head * job->key_stride)
                + (size_t)job->key_begin * HMX_FIXED_K_PADDED;
            if (job->key_begin > 0) {
                qurt_mem_cache_clean(
                    (qurt_addr_t)key,
                    (size_t)(job->n - job->key_begin)
                        * HMX_FIXED_K_PADDED * sizeof(uint16_t),
                    QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            }
            float measured = 0.0f;
            hmx_record_worker_error(
                &job->status,
                hmx_i8_hvx_scale_f16(
                    key, job->n - job->key_begin, HMX_FIXED_K_PADDED,
                    HMX_FIXED_K_PADDED, &measured));
            float *destination = job->scales + job->heads + head;
            if (!job->incremental_key || measured > *destination) {
                *destination = measured;
            }
        }
    }
}

typedef struct hmx_key_pack_job {
    const uint16_t *raw_key;
    int8_t *packed_key;
    int32_t *key_sums;
    const float *head_inverse_k;
    size_t raw_stride;
    size_t packed_stride;
    size_t sums_stride;
    size_t raw_begin;
    size_t raw_bytes;
    size_t packed_prefix;
    size_t sums_prefix;
    size_t packed_begin;
    size_t packed_bytes;
    size_t sums_begin;
    size_t sums_bytes;
    float inverse_k_scale;
    int heads;
    int n;
    int key_begin;
    uint64_t *head_ticks;
    volatile int status;
} hmx_key_pack_job;

static int hmx_pack_key_head(hmx_key_pack_job *job, int head) {
    const uint64_t tick_begin = job->head_ticks != NULL
        ? HAP_perf_get_qtimer_count() : 0;
    int8_t *packed = job->packed_key + (size_t)head * job->packed_stride;
    int32_t *sums = (int32_t *)((uint8_t *)job->key_sums
        + (size_t)head * job->sums_stride);
    const uint16_t *raw = (const uint16_t *)((const uint8_t *)job->raw_key
        + (size_t)head * job->raw_stride);
    if (job->packed_prefix > 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)packed, job->packed_prefix,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        qurt_mem_cache_clean(
            (qurt_addr_t)sums, job->sums_prefix,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    if (job->raw_bytes > 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)((const uint8_t *)raw + job->raw_begin),
            job->raw_bytes, QURT_MEM_CACHE_INVALIDATE,
            QURT_MEM_DCACHE);
    }
    const float inverse_scale = job->head_inverse_k != NULL
        ? job->head_inverse_k[head] : job->inverse_k_scale;
    const int status = hmx_i8_hvx_pack_key_f16_wh(
        packed, sums, raw, job->n, HMX_FIXED_K_PADDED,
        job->key_begin, inverse_scale);
    hmx_record_worker_error(&job->status, status);
    if (status == 0) {
        if (job->packed_bytes > 0) {
            qurt_mem_cache_clean(
                (qurt_addr_t)(packed + job->packed_begin),
                job->packed_bytes, QURT_MEM_CACHE_FLUSH,
                QURT_MEM_DCACHE);
        }
        if (job->sums_bytes > 0) {
            qurt_mem_cache_clean(
                (qurt_addr_t)((uint8_t *)sums + job->sums_begin),
                job->sums_bytes, QURT_MEM_CACHE_FLUSH,
                QURT_MEM_DCACHE);
        }
    }
    if (job->head_ticks != NULL) {
        job->head_ticks[head] =
            HAP_perf_get_qtimer_count() - tick_begin;
    }
    return status;
}

static void hmx_key_pack_worker(void *opaque, int lane, int lanes) {
    hmx_key_pack_job *job = (hmx_key_pack_job *)opaque;
    for (int head = lane; head < job->heads; head += lanes) {
        (void)hmx_pack_key_head(job, head);
    }
}

static void hmx_init_key_pack_job(
    hmx_key_pack_job *job, const uint16_t *raw_key, int8_t *packed_key,
    int32_t *key_sums, int heads, int n, int key_begin,
    float inverse_k_scale, const float *head_inverse_k) {
    memset(job, 0, sizeof(*job));
    job->raw_key = raw_key;
    job->packed_key = packed_key;
    job->key_sums = key_sums;
    job->head_inverse_k = head_inverse_k;
    job->raw_stride = (size_t)n * HMX_FIXED_K_PADDED * sizeof(uint16_t);
    job->packed_stride = (size_t)n * HMX_FIXED_K_PADDED;
    job->sums_stride = (size_t)n * sizeof(int32_t);
    job->raw_begin = (size_t)key_begin * HMX_FIXED_K_PADDED
        * sizeof(uint16_t);
    job->raw_bytes = job->raw_stride - job->raw_begin;
    job->packed_prefix = (size_t)key_begin * HMX_FIXED_K_PADDED;
    job->sums_prefix = (size_t)key_begin * sizeof(int32_t);
    job->packed_begin = (key_begin % 32) == 0
        ? (size_t)key_begin * HMX_FIXED_K_PADDED : 0;
    job->packed_bytes = job->packed_stride - job->packed_begin;
    job->sums_begin = (size_t)key_begin * sizeof(int32_t);
    job->sums_bytes = job->sums_stride - job->sums_begin;
    job->inverse_k_scale = inverse_k_scale;
    job->heads = heads;
    job->n = n;
    job->key_begin = key_begin;
}

static int hmx_parallel_pack_key(
    const uint16_t *raw_key, int8_t *packed_key, int32_t *key_sums,
    int heads, int n, int key_begin, float inverse_k_scale,
    const float *head_inverse_k) {
    hmx_key_pack_job job;
    hmx_init_key_pack_job(
        &job, raw_key, packed_key, key_sums, heads, n, key_begin,
        inverse_k_scale, head_inverse_k);
    if (hmx_hvx_parallel_run(hmx_key_pack_worker, &job) != 0) return -1;
    return job.status;
}

typedef struct hmx_query_pack_job {
    const float *raw_query;
    uint8_t *packed_query;
    const float *head_inverse_q;
    size_t raw_stride;
    size_t packed_stride;
    float inverse_q_scale;
    int heads;
    int query_rows;
    uint64_t *head_ticks;
    volatile int next_head;
    volatile int status;
    qurt_sem_t *ready;
} hmx_query_pack_job;

static int hmx_pack_query_head(hmx_query_pack_job *job, int head) {
    const uint64_t tick_begin = job->head_ticks != NULL
        ? HAP_perf_get_qtimer_count() : 0;
    const float *raw = (const float *)((const uint8_t *)job->raw_query
        + (size_t)head * job->raw_stride);
    uint8_t *packed = job->packed_query + (size_t)head * job->packed_stride;
    qurt_mem_cache_clean(
        (qurt_addr_t)raw, job->raw_stride,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    const float inverse_scale = job->head_inverse_q != NULL
        ? job->head_inverse_q[head] : job->inverse_q_scale;
    const int status = hmx_i8_hvx_pack_query_f32_ah(
        packed, raw, job->query_rows, HMX_FIXED_M_PADDED,
        HMX_FIXED_K_PADDED, inverse_scale);
    hmx_record_worker_error(&job->status, status);
    if (status == 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)packed, job->packed_stride,
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    if (job->head_ticks != NULL) {
        job->head_ticks[head] =
            HAP_perf_get_qtimer_count() - tick_begin;
    }
    return status;
}

static void hmx_query_pack_ahead_worker(void *opaque, int lane, int lanes) {
    (void)lane;
    (void)lanes;
    hmx_query_pack_job *job = (hmx_query_pack_job *)opaque;
    for (;;) {
        const int head = __sync_fetch_and_add(&job->next_head, 1);
        if (head >= job->heads) break;
        (void)hmx_pack_query_head(job, head);
        __asm__ __volatile__("barrier" : : : "memory");
        (void)qurt_sem_up(&job->ready[head]);
    }
}

static void hmx_query_pack_worker(void *opaque, int lane, int lanes) {
    hmx_query_pack_job *job = (hmx_query_pack_job *)opaque;
    for (int head = lane; head < job->heads; head += lanes) {
        (void)hmx_pack_query_head(job, head);
    }
}

typedef struct hmx_cache_heads_job {
    uint8_t *base;
    size_t stride;
    size_t bytes;
    int heads;
    int operation;
} hmx_cache_heads_job;

static void hmx_cache_heads_worker(void *opaque, int lane, int lanes) {
    hmx_cache_heads_job *job = (hmx_cache_heads_job *)opaque;
    for (int head = lane; head < job->heads; head += lanes) {
        qurt_mem_cache_clean(
            (qurt_addr_t)(job->base + (size_t)head * job->stride),
            job->bytes, job->operation, QURT_MEM_DCACHE);
    }
}

static void hmx_parallel_cache_heads(
    void *base, size_t stride, size_t bytes, int heads, int operation) {
    hmx_cache_heads_job job = {
        (uint8_t *)base, stride, bytes, heads, operation};
    (void)hmx_hvx_parallel_run(hmx_cache_heads_worker, &job);
}

static int hmx_pack_query_execute_pipeline(
    const float *raw_query, uint8_t *packed_query,
    const int8_t *packed_key, const int32_t *key_sums, int8_t *output,
    int heads, int query_rows, int n, float inverse_q_scale,
    float requant_scale, const float *head_inverse_q,
    const float *head_requant, hmx_runtime_session *session) {
    const size_t raw_query_stride = (size_t)HMX_FIXED_M_PADDED
        * HMX_FIXED_K_PADDED * sizeof(float);
    const size_t query_stride =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_K_PADDED;
    const size_t key_stride = (size_t)n * HMX_FIXED_K_PADDED;
    const size_t sums_stride = (size_t)n * sizeof(int32_t);
    const size_t output_stride = (size_t)HMX_FIXED_M_PADDED * (size_t)n;
    hmx_query_pack_job job;
    memset(&job, 0, sizeof(job));
    job.raw_query = raw_query;
    job.packed_query = packed_query;
    job.head_inverse_q = head_inverse_q;
    job.raw_stride = raw_query_stride;
    job.packed_stride = query_stride;
    job.inverse_q_scale = inverse_q_scale;
    job.heads = heads;
    job.query_rows = query_rows;

    int status = 0;
    if (heads > 1 && hmx_hvx_worker_pool_lanes() > 1) {
        qurt_sem_t ready[HMX_FIXED_HEADS];
        for (int head = 0; head < heads; ++head) {
            qurt_sem_init_val(&ready[head], 0);
        }
        job.ready = ready;
        job.next_head = 1;
        const int workers_started = hmx_hvx_workers_start(
            hmx_query_pack_ahead_worker, &job) == 0;
        if (workers_started) {
            (void)hmx_pack_query_head(&job, 0);
            for (int head = 0; head < heads; ++head) {
                if (head > 0) (void)qurt_sem_down(&ready[head]);
                if (job.status != 0) {
                    status = job.status;
                    continue;
                }
                status = hmx_i8_kernel_qk_i8_packed(
                    output + (size_t)head * output_stride,
                    packed_query + (size_t)head * query_stride,
                    packed_key + (size_t)head * key_stride,
                    (const int32_t *)((const uint8_t *)key_sums
                                      + (size_t)head * sums_stride),
                    HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n,
                    head_requant != NULL ? head_requant[head] : requant_scale,
                    session->vtcm_base, session->vtcm_size);
                if (status != 0) {
                    hmx_record_worker_error(&job.status, status);
                }
            }
            (void)hmx_hvx_workers_wait();
            if (status == 0) status = job.status;
        }
        for (int head = 0; head < heads; ++head) {
            qurt_sem_destroy(&ready[head]);
        }
        if (workers_started) return status;
        memset(&job, 0, sizeof(job));
        job.raw_query = raw_query;
        job.packed_query = packed_query;
        job.head_inverse_q = head_inverse_q;
        job.raw_stride = raw_query_stride;
        job.packed_stride = query_stride;
        job.inverse_q_scale = inverse_q_scale;
        job.heads = heads;
        job.query_rows = query_rows;
    }

    for (int head = 0; head < heads; ++head) {
        status = hmx_pack_query_head(&job, head);
        if (status != 0) break;
        status = hmx_i8_kernel_qk_i8_packed(
            output + (size_t)head * output_stride,
            packed_query + (size_t)head * query_stride,
            packed_key + (size_t)head * key_stride,
            (const int32_t *)((const uint8_t *)key_sums
                              + (size_t)head * sums_stride),
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n,
            head_requant != NULL ? head_requant[head] : requant_scale,
            session->vtcm_base, session->vtcm_size);
        if (status != 0) break;
    }
    return status;
}

typedef struct hmx_raw_head_pipeline_job {
    hmx_key_pack_job key;
    hmx_query_pack_job query;
    volatile int next_head;
    qurt_sem_t *ready;
} hmx_raw_head_pipeline_job;

#if HMX_HVX_EPILOGUE_LANES > 1
enum { HMX_MIXED_TASK_QUEUE_CAPACITY = 64 };

typedef struct hmx_mixed_task {
    hmx_i8_async_task_fn function;
    void *context;
} hmx_mixed_task;

typedef struct hmx_mixed_task_queue {
    hmx_mixed_task high[HMX_MIXED_TASK_QUEUE_CAPACITY];
    hmx_mixed_task normal[HMX_MIXED_TASK_QUEUE_CAPACITY];
    int high_head;
    int high_tail;
    int high_count;
    int normal_head;
    int normal_tail;
    int normal_count;
    int workers;
    volatile int stopping;
    qurt_mutex_t mutex;
    qurt_sem_t available;
} hmx_mixed_task_queue;

typedef struct hmx_head_pack_task {
    hmx_raw_head_pipeline_job *job;
    int head;
} hmx_head_pack_task;

static void hmx_mixed_task_queue_init(
    hmx_mixed_task_queue *queue, int workers) {
    memset(queue, 0, sizeof(*queue));
    queue->workers = workers;
    qurt_mutex_init(&queue->mutex);
    qurt_sem_init_val(&queue->available, 0);
}

static void hmx_mixed_task_queue_destroy(hmx_mixed_task_queue *queue) {
    qurt_sem_destroy(&queue->available);
    qurt_mutex_destroy(&queue->mutex);
}

static int hmx_mixed_task_queue_push(
    hmx_mixed_task_queue *queue, int high,
    hmx_i8_async_task_fn function, void *context) {
    int status = 0;
    qurt_mutex_lock(&queue->mutex);
    if (queue->stopping) {
        status = -1;
    } else if (high) {
        if (queue->high_count == HMX_MIXED_TASK_QUEUE_CAPACITY) {
            status = -1;
        } else {
            hmx_mixed_task *task = &queue->high[queue->high_tail];
            task->function = function;
            task->context = context;
            queue->high_tail = (queue->high_tail + 1)
                % HMX_MIXED_TASK_QUEUE_CAPACITY;
            ++queue->high_count;
        }
    } else if (queue->normal_count == HMX_MIXED_TASK_QUEUE_CAPACITY) {
        status = -1;
    } else {
        hmx_mixed_task *task = &queue->normal[queue->normal_tail];
        task->function = function;
        task->context = context;
        queue->normal_tail = (queue->normal_tail + 1)
            % HMX_MIXED_TASK_QUEUE_CAPACITY;
        ++queue->normal_count;
    }
    qurt_mutex_unlock(&queue->mutex);
    if (status == 0) (void)qurt_sem_up(&queue->available);
    return status;
}

static int hmx_mixed_task_submit_high(
    void *opaque, hmx_i8_async_task_fn function, void *context) {
    return hmx_mixed_task_queue_push(
        (hmx_mixed_task_queue *)opaque, 1, function, context);
}

static void hmx_mixed_task_worker(void *opaque, int lane, int lanes) {
    (void)lane;
    (void)lanes;
    hmx_mixed_task_queue *queue = (hmx_mixed_task_queue *)opaque;
    for (;;) {
        (void)qurt_sem_down(&queue->available);
        hmx_mixed_task task;
        memset(&task, 0, sizeof(task));
        int stop = 0;
        qurt_mutex_lock(&queue->mutex);
        if (queue->high_count > 0) {
            task = queue->high[queue->high_head];
            queue->high_head = (queue->high_head + 1)
                % HMX_MIXED_TASK_QUEUE_CAPACITY;
            --queue->high_count;
        } else if (queue->normal_count > 0) {
            task = queue->normal[queue->normal_head];
            queue->normal_head = (queue->normal_head + 1)
                % HMX_MIXED_TASK_QUEUE_CAPACITY;
            --queue->normal_count;
        } else if (queue->stopping) {
            stop = 1;
        }
        qurt_mutex_unlock(&queue->mutex);
        if (stop) break;
        if (task.function != NULL) task.function(task.context);
    }
}

static void hmx_mixed_task_queue_stop(hmx_mixed_task_queue *queue) {
    qurt_mutex_lock(&queue->mutex);
    queue->stopping = 1;
    qurt_mutex_unlock(&queue->mutex);
    __asm__ __volatile__("barrier" : : : "memory");
    for (int worker = 0; worker < queue->workers; ++worker) {
        (void)qurt_sem_up(&queue->available);
    }
}

static void hmx_head_pack_task_run(void *opaque) {
    hmx_head_pack_task *task = (hmx_head_pack_task *)opaque;
    hmx_raw_head_pipeline_job *job = task->job;
    if (hmx_pack_key_head(&job->key, task->head) == 0) {
        (void)hmx_pack_query_head(&job->query, task->head);
    }
    __asm__ __volatile__("barrier" : : : "memory");
    (void)qurt_sem_up(&job->ready[task->head]);
}
#endif

static void hmx_raw_head_ahead_worker(void *opaque, int lane, int lanes) {
    (void)lane;
    (void)lanes;
    hmx_raw_head_pipeline_job *job = (hmx_raw_head_pipeline_job *)opaque;
    for (;;) {
        const int head = __sync_fetch_and_add(&job->next_head, 1);
        if (head >= job->query.heads) break;
        if (hmx_pack_key_head(&job->key, head) == 0) {
            (void)hmx_pack_query_head(&job->query, head);
        }
        __asm__ __volatile__("barrier" : : : "memory");
        (void)qurt_sem_up(&job->ready[head]);
    }
}

/* One head is the dependency unit. Persistent HVX workers prepare the K
 * suffix and Q tile for future heads while the FastRPC handler drives HMX
 * (including its HVX readout) for the current head. This removes the old
 * all-head K-pack barrier without changing bucket scales or packed layouts. */
static int hmx_pack_key_query_execute_pipeline(
    const float *raw_query, uint8_t *packed_query,
    const uint16_t *raw_key, int8_t *packed_key, int32_t *key_sums,
    int8_t *output, int heads, int query_rows, int n, int key_begin,
    float inverse_q_scale, float inverse_k_scale, float requant_scale,
    const float *head_inverse_q, const float *head_inverse_k,
    const float *head_requant, hmx_runtime_session *session,
    hmx_i8_dsp_timing *timing) {
    const size_t raw_query_stride = (size_t)HMX_FIXED_M_PADDED
        * HMX_FIXED_K_PADDED * sizeof(float);
    const size_t query_stride =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_K_PADDED;
    const size_t key_stride = (size_t)n * HMX_FIXED_K_PADDED;
    const size_t sums_stride = (size_t)n * sizeof(int32_t);
    const size_t output_stride = (size_t)HMX_FIXED_M_PADDED * (size_t)n;
    hmx_raw_head_pipeline_job job;
    uint64_t key_ticks[HMX_FIXED_HEADS] = {0};
    uint64_t query_ticks[HMX_FIXED_HEADS] = {0};
    memset(&job, 0, sizeof(job));
    hmx_init_key_pack_job(
        &job.key, raw_key, packed_key, key_sums, heads, n, key_begin,
        inverse_k_scale, head_inverse_k);
    job.query.raw_query = raw_query;
    job.query.packed_query = packed_query;
    job.query.head_inverse_q = head_inverse_q;
    job.query.raw_stride = raw_query_stride;
    job.query.packed_stride = query_stride;
    job.query.inverse_q_scale = inverse_q_scale;
    job.query.heads = heads;
    job.query.query_rows = query_rows;
    job.key.head_ticks = key_ticks;
    job.query.head_ticks = query_ticks;

    int status = 0;
    uint64_t hmx_ticks = 0;
    const uint64_t pipeline_begin = HAP_perf_get_qtimer_count();
#if HMX_HVX_EPILOGUE_LANES > 1
    /* One persistent mixed queue replaces the mutually exclusive pack and
     * epilogue pool uses. The handler produces HMX batches. Three workers
     * drain accumulator epilogues first and otherwise pack future heads. */
    if (heads > 1 && hmx_hvx_worker_pool_init() == 0
        && hmx_hvx_worker_pool_lanes() > 1) {
        qurt_sem_t ready[HMX_FIXED_HEADS];
        hmx_head_pack_task pack_tasks[HMX_FIXED_HEADS];
        const int lanes = hmx_hvx_worker_pool_lanes()
            < HMX_HVX_EPILOGUE_LANES
            ? hmx_hvx_worker_pool_lanes() : HMX_HVX_EPILOGUE_LANES;
        hmx_mixed_task_queue queue;
        hmx_mixed_task_queue_init(&queue, lanes - 1);
        for (int head = 0; head < heads; ++head) {
            qurt_sem_init_val(&ready[head], 0);
            pack_tasks[head].job = &job;
            pack_tasks[head].head = head;
        }
        job.ready = ready;
        for (int head = 1; head < heads; ++head) {
            if (hmx_mixed_task_queue_push(
                    &queue, 0, hmx_head_pack_task_run,
                    &pack_tasks[head]) != 0) {
                status = -1;
                break;
            }
        }
        const int workers_started = status == 0
            && hmx_hvx_workers_start(hmx_mixed_task_worker, &queue) == 0;
        if (workers_started) {
            if (hmx_pack_key_head(&job.key, 0) == 0) {
                (void)hmx_pack_query_head(&job.query, 0);
            }
            hmx_i8_task_scheduler scheduler;
            scheduler.context = &queue;
            scheduler.submit_high = hmx_mixed_task_submit_high;
            scheduler.lanes = lanes;
            for (int head = 0; head < heads; ++head) {
                if (head > 0) (void)qurt_sem_down(&ready[head]);
                if (job.key.status != 0 || job.query.status != 0) {
                    status = job.key.status != 0
                        ? job.key.status : job.query.status;
                    continue;
                }
                const uint64_t hmx_begin = HAP_perf_get_qtimer_count();
                status = hmx_i8_kernel_qk_i8_packed_scheduled(
                    output + (size_t)head * output_stride,
                    packed_query + (size_t)head * query_stride,
                    packed_key + (size_t)head * key_stride,
                    (const int32_t *)((const uint8_t *)key_sums
                                      + (size_t)head * sums_stride),
                    HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n,
                    head_requant != NULL
                        ? head_requant[head] : requant_scale,
                    session->vtcm_base, session->vtcm_size, &scheduler);
                hmx_ticks += HAP_perf_get_qtimer_count() - hmx_begin;
                if (status != 0) {
                    hmx_record_worker_error(&job.query.status, status);
                }
            }
            hmx_mixed_task_queue_stop(&queue);
            (void)hmx_hvx_workers_wait();
            if (status == 0) {
                status = job.key.status != 0
                    ? job.key.status : job.query.status;
            }
        } else {
            /* No task ran when the persistent pool could not be acquired. */
            status = 0;
        }
        for (int head = 0; head < heads; ++head) {
            qurt_sem_destroy(&ready[head]);
        }
        hmx_mixed_task_queue_destroy(&queue);
        if (workers_started) {
            if (timing != NULL) {
                timing->pipeline_ticks =
                    HAP_perf_get_qtimer_count() - pipeline_begin;
                timing->hmx_kernel_ticks = hmx_ticks;
                for (int head = 0; head < heads; ++head) {
                    timing->key_pack_work_ticks += key_ticks[head];
                    timing->query_pack_work_ticks += query_ticks[head];
                }
            }
            return status;
        }
    }
#endif
    if (heads > 1 && hmx_hvx_worker_pool_lanes() > 1) {
        qurt_sem_t ready[HMX_FIXED_HEADS];
        for (int head = 0; head < heads; ++head) {
            qurt_sem_init_val(&ready[head], 0);
        }
        job.ready = ready;
        job.next_head = 1;
        const int workers_started = hmx_hvx_workers_start(
            hmx_raw_head_ahead_worker, &job) == 0;
        if (workers_started) {
            if (hmx_pack_key_head(&job.key, 0) == 0) {
                (void)hmx_pack_query_head(&job.query, 0);
            }
            for (int head = 0; head < heads; ++head) {
                if (head > 0) (void)qurt_sem_down(&ready[head]);
                if (job.key.status != 0 || job.query.status != 0) {
                    status = job.key.status != 0
                        ? job.key.status : job.query.status;
                    continue;
                }
                const uint64_t hmx_begin = HAP_perf_get_qtimer_count();
                status = hmx_i8_kernel_qk_i8_packed(
                    output + (size_t)head * output_stride,
                    packed_query + (size_t)head * query_stride,
                    packed_key + (size_t)head * key_stride,
                    (const int32_t *)((const uint8_t *)key_sums
                                      + (size_t)head * sums_stride),
                    HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n,
                    head_requant != NULL
                        ? head_requant[head] : requant_scale,
                    session->vtcm_base, session->vtcm_size);
                hmx_ticks += HAP_perf_get_qtimer_count() - hmx_begin;
                if (status != 0) {
                    hmx_record_worker_error(&job.query.status, status);
                }
            }
            (void)hmx_hvx_workers_wait();
            if (status == 0) {
                status = job.key.status != 0
                    ? job.key.status : job.query.status;
            }
        }
        for (int head = 0; head < heads; ++head) {
            qurt_sem_destroy(&ready[head]);
        }
        if (workers_started) {
            if (timing != NULL) {
                timing->pipeline_ticks =
                    HAP_perf_get_qtimer_count() - pipeline_begin;
                timing->hmx_kernel_ticks = hmx_ticks;
                for (int head = 0; head < heads; ++head) {
                    timing->key_pack_work_ticks += key_ticks[head];
                    timing->query_pack_work_ticks += query_ticks[head];
                }
            }
            return status;
        }
    }

    for (int head = 0; head < heads; ++head) {
        status = hmx_pack_key_head(&job.key, head);
        if (status != 0) break;
        status = hmx_pack_query_head(&job.query, head);
        if (status != 0) break;
        const uint64_t hmx_begin = HAP_perf_get_qtimer_count();
        status = hmx_i8_kernel_qk_i8_packed(
            output + (size_t)head * output_stride,
            packed_query + (size_t)head * query_stride,
            packed_key + (size_t)head * key_stride,
            (const int32_t *)((const uint8_t *)key_sums
                              + (size_t)head * sums_stride),
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n,
            head_requant != NULL ? head_requant[head] : requant_scale,
            session->vtcm_base, session->vtcm_size);
        hmx_ticks += HAP_perf_get_qtimer_count() - hmx_begin;
        if (status != 0) break;
    }
    if (timing != NULL) {
        timing->pipeline_ticks =
            HAP_perf_get_qtimer_count() - pipeline_begin;
        timing->hmx_kernel_ticks = hmx_ticks;
        for (int head = 0; head < heads; ++head) {
            timing->key_pack_work_ticks += key_ticks[head];
            timing->query_pack_work_ticks += query_ticks[head];
        }
    }
    return status;
}

static hmx_registered_arena *find_registered_arena(int32 buffer_fd) {
    for (int index = 0; index < HMX_MAX_REGISTERED_ARENAS; ++index) {
        if (s_registered_arenas[index].users > 0 &&
            s_registered_arenas[index].buffer_fd == buffer_fd) {
            return &s_registered_arenas[index];
        }
    }
    return NULL;
}

static int acquire_arena(
    int32 buffer_fd, uint8_t **buffer, int *temporary_mapping) {
    if (buffer_fd < 0 || !buffer || !temporary_mapping) return -1;
    hmx_registered_arena *registered = find_registered_arena(buffer_fd);
    if (registered) {
        *buffer = registered->buffer;
        *temporary_mapping = 0;
        return registered->buffer ? 0 : -1;
    }
    *buffer = NULL;
    *temporary_mapping = 1;
    return HAP_mmap_get(buffer_fd, (void **)buffer, NULL) || !*buffer ? -1 : 0;
}

static void release_arena(int32 buffer_fd, int temporary_mapping) {
    if (temporary_mapping) HAP_mmap_put(buffer_fd);
}

static void release_all_registered_arenas(void) {
    for (int index = 0; index < HMX_MAX_REGISTERED_ARENAS; ++index) {
        if (s_registered_arenas[index].users > 0) {
            HAP_mmap_put(s_registered_arenas[index].buffer_fd);
            memset(&s_registered_arenas[index], 0,
                   sizeof(s_registered_arenas[index]));
        }
    }
}

AEEResult hmx_int8_rpc_open(const char *uri, remote_handle64 *handle) {
    (void)uri;
    if (!handle) {
        return AEE_EBADPARM;
    }
    ++s_open_handles;
    *handle = (remote_handle64)&s_dummy_handle;
    return AEE_SUCCESS;
}

AEEResult hmx_int8_rpc_close(remote_handle64 handle) {
    (void)handle;
    if (s_open_handles > 0) --s_open_handles;
    if (s_open_handles == 0) {
        if (s_packed_session_active) {
            hmx_runtime_end(&s_packed_session);
            s_packed_session_active = 0;
        }
        release_all_registered_arenas();
        hmx_hvx_worker_pool_deinit();
        hmx_runtime_reset();
    }
    return AEE_SUCCESS;
}

AEEResult hmx_int8_rpc_init_backend(remote_handle64 handle) {
    (void)handle;
    /* Construct the persistent HVX workers lazily on the first parallel
     * preprocessing call.  Metadata/direct HMX users do not pay for them. */
    return hmx_runtime_setup() == 0 ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hmx_int8_rpc_register_arena(
    remote_handle64 handle, int32 buffer_fd) {
    (void)handle;
    if (buffer_fd < 0) return AEE_EBADPARM;
    hmx_registered_arena *registered = find_registered_arena(buffer_fd);
    if (registered) {
        ++registered->users;
        return AEE_SUCCESS;
    }
    int free_index = -1;
    for (int index = 0; index < HMX_MAX_REGISTERED_ARENAS; ++index) {
        if (s_registered_arenas[index].users == 0) {
            free_index = index;
            break;
        }
    }
    if (free_index < 0) return AEE_ENOMEMORY;
    uint8_t *buffer = NULL;
    int temporary_mapping = 0;
    if (acquire_arena(buffer_fd, &buffer, &temporary_mapping) != 0) {
        return AEE_EFAILED;
    }
    s_registered_arenas[free_index].buffer_fd = buffer_fd;
    s_registered_arenas[free_index].buffer = buffer;
    s_registered_arenas[free_index].users = 1;
    return AEE_SUCCESS;
}

AEEResult hmx_int8_rpc_unregister_arena(
    remote_handle64 handle, int32 buffer_fd) {
    (void)handle;
    hmx_registered_arena *registered = find_registered_arena(buffer_fd);
    if (!registered) return AEE_EBADPARM;
    if (--registered->users == 0) {
        HAP_mmap_put(buffer_fd);
        memset(registered, 0, sizeof(*registered));
    }
    return AEE_SUCCESS;
}

AEEResult hmx_int8_rpc_get_fixed_shape(
    remote_handle64 handle,
    int32 *m,
    int32 *k,
    int32 *n) {
    (void)handle;
    if (!m || !k || !n) {
        return AEE_EBADPARM;
    }
    *m = HMX_FIXED_M;
    *k = HMX_FIXED_K;
    *n = HMX_FIXED_N;
    return AEE_SUCCESS;
}

AEEResult hmx_int8_rpc_get_fixed_heads(
    remote_handle64 handle,
    int32 *heads) {
    (void)handle;
    if (!heads) {
        return AEE_EBADPARM;
    }
    *heads = HMX_FIXED_HEADS;
    return AEE_SUCCESS;
}

AEEResult hmx_int8_rpc_get_fixed_scales(
    remote_handle64 handle,
    float *q_scale,
    float *k_scale,
    float *output_scale) {
    (void)handle;
    if (!q_scale || !k_scale || !output_scale) {
        return AEE_EBADPARM;
    }
#if defined(HMX_OPERATOR_Q_SCALE) && defined(HMX_OPERATOR_K_SCALE) && \
    defined(HMX_OPERATOR_OUTPUT_SCALE)
    *q_scale = (float)(HMX_OPERATOR_Q_SCALE);
    *k_scale = (float)(HMX_OPERATOR_K_SCALE);
    *output_scale = (float)(HMX_OPERATOR_OUTPUT_SCALE);
    return AEE_SUCCESS;
#else
    *q_scale = 0.0f;
    *k_scale = 0.0f;
    *output_scale = 0.0f;
    return AEE_EFAILED;
#endif
}

static AEEResult run_matmul(
    int32 buffer_fd,
    int32 lhs_offset,
    int32 rhs_offset,
    int32 output_offset,
    int32 m,
    int32 k,
    int32 n,
    int fixed_shape,
    int output_i8,
    float output_scale) {
    if (buffer_fd < 0 || lhs_offset < 0 || rhs_offset < 0 || output_offset < 0 ||
        m <= 0 || k <= 0 || n <= 0 || (m % 64) || (k % 32) || (n % 32) ||
        (output_i8 &&
         (!fixed_shape || !(output_scale >= 0.0f) || output_scale > FLT_MAX))) {
        return AEE_EBADPARM;
    }

    uint8_t *buffer = NULL;
    int err = HAP_mmap_get(buffer_fd, (void **)&buffer, NULL);
    if (err || !buffer) {
        FARF(ALWAYS, "hmx_int8: HAP_mmap_get failed 0x%x", err);
        return AEE_EFAILED;
    }

    uint8_t *lhs = buffer + lhs_offset;
    int8_t *rhs = (int8_t *)(buffer + rhs_offset);
    void *output = buffer + output_offset;
    size_t lhs_bytes = (size_t)m * (size_t)k;
    size_t rhs_bytes = (size_t)k * (size_t)n;
    size_t output_bytes =
        (size_t)m * (size_t)n * (output_i8 ? sizeof(int8_t) : sizeof(int32_t));

    qurt_mem_cache_clean(
        (qurt_addr_t)lhs, lhs_bytes, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    qurt_mem_cache_clean(
        (qurt_addr_t)rhs, rhs_bytes, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    int kernel_status = -1;
    hmx_runtime_session session;
    if (hmx_runtime_begin(&session, 65536u) == 0) {
        if (output_i8) {
            kernel_status = hmx_i8_kernel_i8_fixed(
                (int8_t *)output,
                lhs,
                rhs,
                output_scale,
                session.vtcm_base,
                session.vtcm_size);
        } else if (fixed_shape) {
            kernel_status = hmx_i8_kernel_fixed(
                (int32_t *)output,
                lhs,
                rhs,
                session.vtcm_base,
                session.vtcm_size);
        } else {
            kernel_status = hmx_i8_kernel(
                (int32_t *)output,
                lhs,
                rhs,
                m,
                k,
                n,
                session.vtcm_base,
                session.vtcm_size);
        }
        hmx_runtime_end(&session);
    }

    if (kernel_status == 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)output, output_bytes, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    HAP_mmap_put(buffer_fd);
    return kernel_status == 0 ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hmx_int8_rpc_matmul(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 lhs_offset,
    int32 rhs_offset,
    int32 output_offset,
    int32 m,
    int32 k,
    int32 n) {
    (void)handle;
    return run_matmul(
        buffer_fd,
        lhs_offset,
        rhs_offset,
        output_offset,
        m,
        k,
        n,
        0,
        0,
        1.0f);
}

AEEResult hmx_int8_rpc_matmul_fixed(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 lhs_offset,
    int32 rhs_offset,
    int32 output_offset) {
    (void)handle;
    return run_matmul(
        buffer_fd,
        lhs_offset,
        rhs_offset,
        output_offset,
        HMX_FIXED_M_PADDED,
        HMX_FIXED_K_PADDED,
        HMX_FIXED_N_PADDED,
        1,
        0,
        1.0f);
}

AEEResult hmx_int8_rpc_matmul_i8_fixed(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 lhs_offset,
    int32 rhs_offset,
    int32 output_offset,
    float output_scale) {
    (void)handle;
    return run_matmul(
        buffer_fd,
        lhs_offset,
        rhs_offset,
        output_offset,
        HMX_FIXED_M_PADDED,
        HMX_FIXED_K_PADDED,
        HMX_FIXED_N_PADDED,
        1,
        1,
        output_scale);
}

AEEResult hmx_int8_rpc_matmul_qk_i8_fixed(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 query_offset,
    int32 key_offset,
    int32 output_offset) {
    (void)handle;
    if (buffer_fd < 0 || query_offset < 0 || key_offset < 0 ||
        output_offset < 0) {
        return AEE_EBADPARM;
    }

#if defined(HMX_OPERATOR_Q_SCALE) && defined(HMX_OPERATOR_K_SCALE) && \
    defined(HMX_OPERATOR_OUTPUT_SCALE)
    uint8_t *buffer = NULL;
    int err = HAP_mmap_get(buffer_fd, (void **)&buffer, NULL);
    if (err || !buffer) {
        FARF(ALWAYS, "hmx_int8: QK HAP_mmap_get failed 0x%x", err);
        return AEE_EFAILED;
    }

    uint8_t *query = buffer + query_offset;
    int8_t *key = (int8_t *)(buffer + key_offset);
    int8_t *output = (int8_t *)(buffer + output_offset);
    const size_t query_stride =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_K_PADDED;
    const size_t key_stride =
        (size_t)HMX_FIXED_N_PADDED * HMX_FIXED_K_PADDED;
    const size_t output_stride =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_N_PADDED;

    qurt_mem_cache_clean(
        (qurt_addr_t)query,
        query_stride * HMX_FIXED_HEADS,
        QURT_MEM_CACHE_INVALIDATE,
        QURT_MEM_DCACHE);
    qurt_mem_cache_clean(
        (qurt_addr_t)key,
        key_stride * HMX_FIXED_HEADS,
        QURT_MEM_CACHE_INVALIDATE,
        QURT_MEM_DCACHE);

    const float requant_scale =
        (float)(HMX_OPERATOR_Q_SCALE) * (float)(HMX_OPERATOR_K_SCALE) /
        (float)(HMX_OPERATOR_OUTPUT_SCALE);
    int kernel_status = -1;
    hmx_runtime_session session;
    if (hmx_runtime_begin(&session, 65536u) == 0) {
        kernel_status = 0;
        for (int32 head = 0; head < HMX_FIXED_HEADS; ++head) {
            kernel_status = hmx_i8_kernel_qk_i8_fixed(
                output + (size_t)head * output_stride,
                query + (size_t)head * query_stride,
                key + (size_t)head * key_stride,
                requant_scale,
                session.vtcm_base,
                session.vtcm_size);
            if (kernel_status != 0) {
                break;
            }
        }
        hmx_runtime_end(&session);
    }

    if (kernel_status == 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)output,
            output_stride * HMX_FIXED_HEADS,
            QURT_MEM_CACHE_FLUSH,
            QURT_MEM_DCACHE);
    }
    HAP_mmap_put(buffer_fd);
    return kernel_status == 0 ? AEE_SUCCESS : AEE_EFAILED;
#else
    return AEE_EFAILED;
#endif
}

AEEResult hmx_int8_rpc_execute_qk_i8_direct(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 query_offset,
    int32 key_offset,
    int32 output_offset,
    int32 heads,
    int32 n) {
    (void)handle;
    if (buffer_fd < 0 || query_offset < 0 || key_offset < 0 ||
        output_offset < 0 || heads <= 0 || heads > HMX_FIXED_HEADS ||
        n <= 0 || n > HMX_FIXED_N_PADDED || (n % 32) != 0) {
        return AEE_EBADPARM;
    }

#if defined(HMX_OPERATOR_Q_SCALE) && defined(HMX_OPERATOR_K_SCALE) && \
    defined(HMX_OPERATOR_OUTPUT_SCALE)
    uint8_t *buffer = NULL;
    int err = HAP_mmap_get(buffer_fd, (void **)&buffer, NULL);
    if (err || !buffer) {
        FARF(ALWAYS, "hmx_int8: direct QK HAP_mmap_get failed 0x%x", err);
        return AEE_EFAILED;
    }

    int8_t *query = (int8_t *)(buffer + query_offset);
    int8_t *key = (int8_t *)(buffer + key_offset);
    int8_t *output = (int8_t *)(buffer + output_offset);
    const size_t query_stride =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_K_PADDED;
    const size_t key_stride = (size_t)n * HMX_FIXED_K_PADDED;
    const size_t output_stride = (size_t)HMX_FIXED_M_PADDED * (size_t)n;
    qurt_mem_cache_clean(
        (qurt_addr_t)query,
        query_stride * (size_t)heads,
        QURT_MEM_CACHE_INVALIDATE,
        QURT_MEM_DCACHE);
    qurt_mem_cache_clean(
        (qurt_addr_t)key,
        key_stride * (size_t)heads,
        QURT_MEM_CACHE_INVALIDATE,
        QURT_MEM_DCACHE);

    const float requant_scale =
        (float)(HMX_OPERATOR_Q_SCALE) * (float)(HMX_OPERATOR_K_SCALE) /
        (float)(HMX_OPERATOR_OUTPUT_SCALE);
    int kernel_status = -1;
    hmx_runtime_session session;
    const uint32_t vtcm_bytes = hmx_i8_qk_direct_vtcm_bytes(
        HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n);
    if (hmx_runtime_begin(&session, vtcm_bytes) == 0) {
        kernel_status = 0;
        for (int32 head = 0; head < heads; ++head) {
            kernel_status = hmx_i8_kernel_qk_i8_direct(
                output + (size_t)head * output_stride,
                query + (size_t)head * query_stride,
                key + (size_t)head * key_stride,
                HMX_FIXED_M_PADDED,
                HMX_FIXED_K_PADDED,
                n,
                requant_scale,
                session.vtcm_base,
                session.vtcm_size);
            if (kernel_status != 0) {
                break;
            }
        }
        hmx_runtime_end(&session);
    }
    if (kernel_status == 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)output,
            output_stride * (size_t)heads,
            QURT_MEM_CACHE_FLUSH,
            QURT_MEM_DCACHE);
    }
    HAP_mmap_put(buffer_fd);
    return kernel_status == 0 ? AEE_SUCCESS : AEE_EFAILED;
#else
    return AEE_EFAILED;
#endif
}

AEEResult hmx_int8_rpc_begin_qk_i8_packed(
    remote_handle64 handle,
    int32 n) {
    (void)handle;
    if (n <= 0 || n > HMX_FIXED_N_PADDED || (n % 32) != 0 ||
        s_packed_session_active) {
        return AEE_EBADPARM;
    }
    const uint32_t vtcm_bytes = hmx_i8_qk_packed_vtcm_bytes(
        HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n);
    if (vtcm_bytes == 0 ||
        hmx_runtime_begin(&s_packed_session, vtcm_bytes) != 0) {
        return AEE_EFAILED;
    }
    s_packed_session_active = 1;
    return AEE_SUCCESS;
}

AEEResult hmx_int8_rpc_end_qk_i8_packed(remote_handle64 handle) {
    (void)handle;
    if (!s_packed_session_active) return AEE_EBADPARM;
    hmx_runtime_end(&s_packed_session);
    s_packed_session_active = 0;
    return AEE_SUCCESS;
}

AEEResult hmx_int8_rpc_execute_qk_i8_packed(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 query_offset,
    int32 key_offset,
    int32 key_sums_offset,
    int32 output_offset,
    int32 heads,
    int32 n,
    float requant_scale) {
    (void)handle;
    if (buffer_fd < 0 || query_offset < 0 || key_offset < 0 ||
        key_sums_offset < 0 || output_offset < 0 || heads <= 0 ||
        heads > HMX_FIXED_HEADS || n <= 0 || n > HMX_FIXED_N_PADDED ||
        (n % 32) != 0 || !(requant_scale >= 0.0f) ||
        requant_scale > FLT_MAX) {
        return AEE_EBADPARM;
    }
    uint8_t *buffer = NULL;
    int err = HAP_mmap_get(buffer_fd, (void **)&buffer, NULL);
    if (err || !buffer) return AEE_EFAILED;

    const size_t query_stride =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_K_PADDED;
    const size_t key_stride = (size_t)n * HMX_FIXED_K_PADDED;
    const size_t sums_stride = (size_t)n * sizeof(int32_t);
    const size_t output_stride = (size_t)HMX_FIXED_M_PADDED * (size_t)n;
    uint8_t *query = buffer + query_offset;
    int8_t *key = (int8_t *)(buffer + key_offset);
    int32_t *key_sums = (int32_t *)(buffer + key_sums_offset);
    int8_t *output = (int8_t *)(buffer + output_offset);
    qurt_mem_cache_clean(
        (qurt_addr_t)query, query_stride * (size_t)heads,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    qurt_mem_cache_clean(
        (qurt_addr_t)key, key_stride * (size_t)heads,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    qurt_mem_cache_clean(
        (qurt_addr_t)key_sums, sums_stride * (size_t)heads,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    hmx_runtime_session temporary_session;
    hmx_runtime_session *session = &s_packed_session;
    int temporary = 0;
    if (!s_packed_session_active) {
        const uint32_t vtcm_bytes = hmx_i8_qk_packed_vtcm_bytes(
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n);
        if (hmx_runtime_begin(&temporary_session, vtcm_bytes) != 0) {
            HAP_mmap_put(buffer_fd);
            return AEE_EFAILED;
        }
        session = &temporary_session;
        temporary = 1;
    }

    int kernel_status = 0;
    for (int32 head = 0; head < heads; ++head) {
        kernel_status = hmx_i8_kernel_qk_i8_packed(
            output + (size_t)head * output_stride,
            query + (size_t)head * query_stride,
            key + (size_t)head * key_stride,
            (const int32_t *)((const uint8_t *)key_sums
                              + (size_t)head * sums_stride),
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n, requant_scale,
            session->vtcm_base, session->vtcm_size);
        if (kernel_status != 0) break;
    }
    if (temporary) hmx_runtime_end(&temporary_session);
    if (kernel_status == 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)output, output_stride * (size_t)heads,
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    HAP_mmap_put(buffer_fd);
    return kernel_status == 0 ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hmx_int8_rpc_execute_qk_i8_cpu_packed_per_head(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 query_offset,
    int32 key_offset,
    int32 key_sums_offset,
    int32 output_offset,
    int32 scales_offset,
    int32 ready_offset,
    int32 heads,
    int32 n) {
    (void)handle;
    if (buffer_fd < 0 || query_offset < 0 || key_offset < 0 ||
        key_sums_offset < 0 || output_offset < 0 || scales_offset < 0 ||
        ready_offset < 0 || heads <= 0 || heads > HMX_FIXED_HEADS ||
        n <= 0 || n > HMX_FIXED_N_PADDED || (n % 32) != 0) {
        return AEE_EBADPARM;
    }

    const uint64_t total_begin = HAP_perf_get_qtimer_count();
    hmx_i8_dsp_timing timing;
    memset(&timing, 0, sizeof(timing));
    timing.struct_size = sizeof(timing);
    timing.version = HMX_I8_DSP_TIMING_VERSION;

    uint8_t *buffer = NULL;
    int temporary_mapping = 0;
    if (acquire_arena(buffer_fd, &buffer, &temporary_mapping) != 0) {
        return AEE_EFAILED;
    }

    const size_t query_stride =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_K_PADDED;
    const size_t key_stride = (size_t)n * HMX_FIXED_K_PADDED;
    const size_t sums_stride = (size_t)n * sizeof(int32_t);
    const size_t output_stride = (size_t)HMX_FIXED_M_PADDED * (size_t)n;
    uint8_t *query = buffer + query_offset;
    int8_t *key = (int8_t *)(buffer + key_offset);
    int32_t *key_sums = (int32_t *)(buffer + key_sums_offset);
    int8_t *output = (int8_t *)(buffer + output_offset);
    float *scale_data = (float *)(buffer + scales_offset);
    const float *head_requant = scale_data + heads * 2;
    volatile int32_t *ready =
        (volatile int32_t *)(buffer + ready_offset);
    qurt_mem_cache_clean(
        (qurt_addr_t)head_requant, (size_t)heads * sizeof(float),
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    for (int32 head = 0; head < heads; ++head) {
        if (!(head_requant[head] >= 0.0f) ||
            head_requant[head] > FLT_MAX) {
            release_arena(buffer_fd, temporary_mapping);
            return AEE_EBADPARM;
        }
    }

    hmx_runtime_session temporary_session;
    hmx_runtime_session *session = &s_packed_session;
    int temporary_session_active = 0;
    int status = 0;
    if (!s_packed_session_active) {
        const uint32_t vtcm_bytes = hmx_i8_qk_packed_vtcm_bytes(
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n);
        const uint64_t resource_begin = HAP_perf_get_qtimer_count();
        if (hmx_runtime_begin(&temporary_session, vtcm_bytes) != 0) {
            status = -1;
        } else {
            session = &temporary_session;
            temporary_session_active = 1;
        }
        timing.resource_begin_ticks =
            HAP_perf_get_qtimer_count() - resource_begin;
    }

    const uint64_t pipeline_begin = HAP_perf_get_qtimer_count();
    int32 publish_begin = 0;
    for (int32 head = 0; head < heads && status == 0; ++head) {
        volatile int32_t *head_ready = ready
            + (size_t)head * HMX_I8_CPU_PACK_READY_STRIDE;
        const uint64_t wait_begin = HAP_perf_get_qtimer_count();
        int32 producer_state = 0;
        for (;;) {
            qurt_mem_cache_clean(
                (qurt_addr_t)head_ready, sizeof(*head_ready),
                QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            __asm__ __volatile__("barrier" : : : "memory");
            producer_state = __atomic_load_n(
                head_ready, __ATOMIC_ACQUIRE);
            if (producer_state != 0) break;
            /* A pack failure must not strand the FastRPC handler forever. */
            if (HAP_perf_get_qtimer_count() - wait_begin
                    > UINT64_C(5) * UINT64_C(19200000)) {
                FARF(ALWAYS,
                     "hmx_int8: CPU pack timeout waiting for head=%d", head);
                status = -1;
                break;
            }
        }
        if (status != 0 || producer_state < 0) {
            status = -1;
            break;
        }

        const int streamed =
            (producer_state & HMX_I8_CPU_PACK_STREAM_FLAG) != 0;
        if (!streamed) {
            qurt_mem_cache_clean(
                (qurt_addr_t)(query + (size_t)head * query_stride),
                query_stride, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            qurt_mem_cache_clean(
                (qurt_addr_t)(key + (size_t)head * key_stride),
                key_stride, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            qurt_mem_cache_clean(
                (qurt_addr_t)((uint8_t *)key_sums
                              + (size_t)head * sums_stride),
                sums_stride, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        }
        const uint64_t hmx_begin = HAP_perf_get_qtimer_count();
        if (streamed) {
            status = hmx_i8_kernel_qk_i8_packed_streamed(
                output + (size_t)head * output_stride,
                query + (size_t)head * query_stride,
                key + (size_t)head * key_stride,
                (const int32_t *)((const uint8_t *)key_sums
                                  + (size_t)head * sums_stride),
                HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n,
                head_requant[head], head_ready,
                session->vtcm_base, session->vtcm_size);
        } else {
            status = hmx_i8_kernel_qk_i8_packed(
                output + (size_t)head * output_stride,
                query + (size_t)head * query_stride,
                key + (size_t)head * key_stride,
                (const int32_t *)((const uint8_t *)key_sums
                                  + (size_t)head * sums_stride),
                HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n,
                head_requant[head], session->vtcm_base, session->vtcm_size);
        }
        timing.hmx_kernel_ticks +=
            HAP_perf_get_qtimer_count() - hmx_begin;
        if (status == 0
            && (head + 1 == heads
                || head + 1 - publish_begin
                    == HMX_LONG_RPC_READY_GROUP_HEADS)) {
            /* RpcMem is uncached on ARM but cached on the DSP. Publish one
             * contiguous macro-group only after every score block in it has
             * reached DDR. Keeping the HMX loop H12-fused while reducing
             * cache-maintenance and host notification frequency avoids the
             * per-head long-RPC protocol overhead. */
            const int32 publish_count = head + 1 - publish_begin;
            const uint64_t output_flush = HAP_perf_get_qtimer_count();
            qurt_mem_cache_clean(
                (qurt_addr_t)(output
                              + (size_t)publish_begin * output_stride),
                (size_t)publish_count * output_stride,
                QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            timing.output_flush_ticks +=
                HAP_perf_get_qtimer_count() - output_flush;
            __asm__ __volatile__("barrier" : : : "memory");
            for (int32 published = publish_begin;
                 published <= head; ++published) {
                volatile int32_t *published_ready = ready
                    + (size_t)published * HMX_I8_CPU_PACK_READY_STRIDE;
                __atomic_store_n(
                    published_ready, HMX_I8_CPU_PACK_OUTPUT_READY,
                    __ATOMIC_RELEASE);
            }
            volatile int32_t *first_ready = ready
                + (size_t)publish_begin * HMX_I8_CPU_PACK_READY_STRIDE;
            const size_t ready_bytes =
                ((size_t)(publish_count - 1)
                     * HMX_I8_CPU_PACK_READY_STRIDE
                 + 1U)
                * sizeof(*first_ready);
            qurt_mem_cache_clean(
                (qurt_addr_t)first_ready, ready_bytes,
                QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            publish_begin = head + 1;
        }
    }
    timing.pipeline_ticks =
        HAP_perf_get_qtimer_count() - pipeline_begin;

    if (status != 0) {
        /* Wake an ARM poller even when the handler exits before producing all
         * heads. A still-running CPU producer may update a future progress
         * word, but the FastRPC completion remains the authoritative error. */
        for (int32 head = 0; head < heads; ++head) {
            volatile int32_t *head_ready = ready
                + (size_t)head * HMX_I8_CPU_PACK_READY_STRIDE;
            qurt_mem_cache_clean(
                (qurt_addr_t)head_ready, sizeof(*head_ready),
                QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
            if (__atomic_load_n(head_ready, __ATOMIC_ACQUIRE)
                    != HMX_I8_CPU_PACK_OUTPUT_READY) {
                __atomic_store_n(
                    head_ready, HMX_I8_CPU_PACK_OUTPUT_ERROR,
                    __ATOMIC_RELEASE);
                qurt_mem_cache_clean(
                    (qurt_addr_t)head_ready, sizeof(*head_ready),
                    QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
            }
        }
    }

    if (temporary_session_active) {
        const uint64_t resource_end = HAP_perf_get_qtimer_count();
        hmx_runtime_end(&temporary_session);
        timing.resource_end_ticks =
            HAP_perf_get_qtimer_count() - resource_end;
    }
    /* Successful outputs were flushed and published head-by-head above. */
    timing.total_ticks = HAP_perf_get_qtimer_count() - total_begin;
    hmx_i8_dsp_timing *published = (hmx_i8_dsp_timing *)(
        buffer + scales_offset + hmx_i8_dsp_timing_offset(heads));
    *published = timing;
    qurt_mem_cache_clean(
        (qurt_addr_t)published, sizeof(*published),
        QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    release_arena(buffer_fd, temporary_mapping);
    return status == 0 ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hmx_int8_rpc_prepare_qk_i8_raw_key(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 raw_key_offset,
    int32 packed_key_offset,
    int32 key_sums_offset,
    int32 heads,
    int32 n,
    int32 key_begin,
    float inverse_k_scale) {
    (void)handle;
    if (buffer_fd < 0 || raw_key_offset < 0 || packed_key_offset < 0 ||
        key_sums_offset < 0 || heads <= 0 || heads > HMX_FIXED_HEADS ||
        n <= 0 || n > HMX_FIXED_N_PADDED || (n % 32) != 0 ||
        key_begin < 0 || key_begin > n || !(inverse_k_scale > 0.0f) ||
        inverse_k_scale > FLT_MAX) {
        return AEE_EBADPARM;
    }
    uint8_t *buffer = NULL;
    int temporary_mapping = 0;
    if (acquire_arena(buffer_fd, &buffer, &temporary_mapping) != 0) {
        return AEE_EFAILED;
    }
    uint16_t *raw_key = (uint16_t *)(buffer + raw_key_offset);
    int8_t *packed_key = (int8_t *)(buffer + packed_key_offset);
    int32_t *key_sums = (int32_t *)(buffer + key_sums_offset);
    /* Prefix invalidation, suffix quantization/WH packing, K sums, and the
     * publication flush are one four-lane per-head phase. */
    const int status = hmx_parallel_pack_key(
        raw_key, packed_key, key_sums, heads, n, key_begin,
        inverse_k_scale, NULL);
    release_arena(buffer_fd, temporary_mapping);
    return status == 0 ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hmx_int8_rpc_execute_qk_i8_raw_query(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 raw_query_offset,
    int32 packed_query_offset,
    int32 packed_key_offset,
    int32 key_sums_offset,
    int32 output_offset,
    int32 heads,
    int32 query_rows,
    int32 n,
    float inverse_q_scale,
    float requant_scale) {
    (void)handle;
    if (buffer_fd < 0 || raw_query_offset < 0 || packed_query_offset < 0 ||
        packed_key_offset < 0 || key_sums_offset < 0 || output_offset < 0 ||
        heads <= 0 || heads > HMX_FIXED_HEADS || query_rows <= 0 ||
        query_rows > HMX_FIXED_M || n <= 0 || n > HMX_FIXED_N_PADDED ||
        (n % 32) != 0 || !(inverse_q_scale > 0.0f) ||
        inverse_q_scale > FLT_MAX || !(requant_scale >= 0.0f) ||
        requant_scale > FLT_MAX) {
        return AEE_EBADPARM;
    }
    uint8_t *buffer = NULL;
    int temporary_mapping = 0;
    if (acquire_arena(buffer_fd, &buffer, &temporary_mapping) != 0) {
        return AEE_EFAILED;
    }
    const size_t key_stride = (size_t)n * HMX_FIXED_K_PADDED;
    const size_t sums_stride = (size_t)n * sizeof(int32_t);
    const size_t output_stride = (size_t)HMX_FIXED_M_PADDED * (size_t)n;
    float *raw_query = (float *)(buffer + raw_query_offset);
    uint8_t *packed_query = buffer + packed_query_offset;
    int8_t *packed_key = (int8_t *)(buffer + packed_key_offset);
    int32_t *key_sums = (int32_t *)(buffer + key_sums_offset);
    int8_t *output = (int8_t *)(buffer + output_offset);
    hmx_parallel_cache_heads(
        packed_key, key_stride, key_stride, heads,
        QURT_MEM_CACHE_INVALIDATE);
    hmx_parallel_cache_heads(
        key_sums, sums_stride, sums_stride, heads,
        QURT_MEM_CACHE_INVALIDATE);

    hmx_runtime_session temporary_session;
    hmx_runtime_session *session = &s_packed_session;
    int temporary = 0;
    if (!s_packed_session_active) {
        const uint32_t vtcm_bytes = hmx_i8_qk_packed_vtcm_bytes(
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n);
        if (hmx_runtime_begin(&temporary_session, vtcm_bytes) != 0) {
            release_arena(buffer_fd, temporary_mapping);
            return AEE_EFAILED;
        }
        session = &temporary_session;
        temporary = 1;
    }
    const int status = hmx_pack_query_execute_pipeline(
        raw_query, packed_query, packed_key, key_sums, output,
        heads, query_rows, n, inverse_q_scale, requant_scale,
        NULL, NULL, session);
    if (temporary) hmx_runtime_end(&temporary_session);
    if (status == 0) {
        hmx_parallel_cache_heads(
            output, output_stride, output_stride, heads,
            QURT_MEM_CACHE_FLUSH);
    }
    release_arena(buffer_fd, temporary_mapping);
    return status == 0 ? AEE_SUCCESS : AEE_EFAILED;
}

static AEEResult hmx_prepare_execute_qk_i8_raw_impl(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 raw_query_offset,
    int32 raw_key_offset,
    int32 packed_query_offset,
    int32 packed_key_offset,
    int32 key_sums_offset,
    int32 output_offset,
    int32 heads,
    int32 query_rows,
    int32 n,
    int32 key_begin,
    int32 scales_offset,
    int per_head_scales,
    float inverse_q_scale,
    float inverse_k_scale,
    float requant_scale) {
    (void)handle;
    if (buffer_fd < 0 || raw_query_offset < 0 || raw_key_offset < 0 ||
        packed_query_offset < 0 || packed_key_offset < 0 ||
        key_sums_offset < 0 || output_offset < 0 || heads <= 0 ||
        heads > HMX_FIXED_HEADS || query_rows <= 0 ||
        query_rows > HMX_FIXED_M || n <= 0 || n > HMX_FIXED_N_PADDED ||
        (n % 32) != 0 || key_begin < 0 || key_begin > n ||
        (per_head_scales && scales_offset < 0) ||
        (!per_head_scales &&
         (!(inverse_q_scale > 0.0f) || inverse_q_scale > FLT_MAX ||
          !(inverse_k_scale > 0.0f) || inverse_k_scale > FLT_MAX ||
          !(requant_scale >= 0.0f) || requant_scale > FLT_MAX))) {
        return AEE_EBADPARM;
    }

    const uint64_t total_begin = HAP_perf_get_qtimer_count();
    hmx_i8_dsp_timing timing;
    memset(&timing, 0, sizeof(timing));
    timing.struct_size = sizeof(timing);
    timing.version = HMX_I8_DSP_TIMING_VERSION;

    uint8_t *buffer = NULL;
    int temporary_mapping = 0;
    if (acquire_arena(buffer_fd, &buffer, &temporary_mapping) != 0) {
        return AEE_EFAILED;
    }

    const size_t output_stride = (size_t)HMX_FIXED_M_PADDED * (size_t)n;
    float *raw_query = (float *)(buffer + raw_query_offset);
    uint16_t *raw_key = (uint16_t *)(buffer + raw_key_offset);
    uint8_t *packed_query = buffer + packed_query_offset;
    int8_t *packed_key = (int8_t *)(buffer + packed_key_offset);
    int32_t *key_sums = (int32_t *)(buffer + key_sums_offset);
    int8_t *output = (int8_t *)(buffer + output_offset);
    const float *head_inverse_q = NULL;
    const float *head_inverse_k = NULL;
    const float *head_requant = NULL;
    if (per_head_scales) {
        float *scale_data = (float *)(buffer + scales_offset);
        qurt_mem_cache_clean(
            (qurt_addr_t)scale_data, (size_t)heads * 3u * sizeof(float),
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
        head_inverse_q = scale_data;
        head_inverse_k = scale_data + heads;
        head_requant = scale_data + heads * 2;
        for (int32 head = 0; head < heads; ++head) {
            if (!(head_inverse_q[head] > 0.0f) ||
                head_inverse_q[head] > FLT_MAX ||
                !(head_inverse_k[head] > 0.0f) ||
                head_inverse_k[head] > FLT_MAX ||
                !(head_requant[head] >= 0.0f) ||
                head_requant[head] > FLT_MAX) {
                release_arena(buffer_fd, temporary_mapping);
                return AEE_EBADPARM;
            }
        }
    }

    int status = 0;
    hmx_runtime_session temporary_session;
    hmx_runtime_session *session = &s_packed_session;
    int temporary_session_active = 0;
    if (status == 0 && !s_packed_session_active) {
        const uint32_t vtcm_bytes = hmx_i8_qk_packed_vtcm_bytes(
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n);
        const uint64_t resource_begin = HAP_perf_get_qtimer_count();
        if (hmx_runtime_begin(&temporary_session, vtcm_bytes) != 0) {
            status = -1;
        } else {
            session = &temporary_session;
            temporary_session_active = 1;
        }
        timing.resource_begin_ticks =
            HAP_perf_get_qtimer_count() - resource_begin;
    }

    if (status == 0) {
        /* K suffix pack, Q pack, HMX, and the HVX output epilogue form one
         * per-head pipeline. Future heads are published before consumption;
         * no all-head K or Q barrier remains. */
        status = hmx_pack_key_query_execute_pipeline(
            raw_query, packed_query, raw_key, packed_key, key_sums, output,
            heads, query_rows, n, key_begin, inverse_q_scale,
            inverse_k_scale, requant_scale,
            per_head_scales ? head_inverse_q : NULL,
            per_head_scales ? head_inverse_k : NULL,
            per_head_scales ? head_requant : NULL, session, &timing);
    }
    if (temporary_session_active) {
        const uint64_t resource_end = HAP_perf_get_qtimer_count();
        hmx_runtime_end(&temporary_session);
        timing.resource_end_ticks =
            HAP_perf_get_qtimer_count() - resource_end;
    }
    if (status == 0) {
        const uint64_t output_flush = HAP_perf_get_qtimer_count();
        hmx_parallel_cache_heads(
            output, output_stride, output_stride, heads,
            QURT_MEM_CACHE_FLUSH);
        timing.output_flush_ticks =
            HAP_perf_get_qtimer_count() - output_flush;
    }
    timing.total_ticks = HAP_perf_get_qtimer_count() - total_begin;
    if (per_head_scales) {
        hmx_i8_dsp_timing *published = (hmx_i8_dsp_timing *)(
            buffer + scales_offset + hmx_i8_dsp_timing_offset(heads));
        *published = timing;
        qurt_mem_cache_clean(
            (qurt_addr_t)published, sizeof(*published),
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    release_arena(buffer_fd, temporary_mapping);
    return status == 0 ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hmx_int8_rpc_prepare_execute_qk_i8_raw(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 raw_query_offset,
    int32 raw_key_offset,
    int32 packed_query_offset,
    int32 packed_key_offset,
    int32 key_sums_offset,
    int32 output_offset,
    int32 heads,
    int32 query_rows,
    int32 n,
    int32 key_begin,
    float inverse_q_scale,
    float inverse_k_scale,
    float requant_scale) {
    return hmx_prepare_execute_qk_i8_raw_impl(
        handle, buffer_fd, raw_query_offset, raw_key_offset,
        packed_query_offset, packed_key_offset, key_sums_offset,
        output_offset, heads, query_rows, n, key_begin, 0, 0,
        inverse_q_scale, inverse_k_scale, requant_scale);
}

AEEResult hmx_int8_rpc_prepare_execute_qk_i8_raw_per_head(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 raw_query_offset,
    int32 raw_key_offset,
    int32 packed_query_offset,
    int32 packed_key_offset,
    int32 key_sums_offset,
    int32 output_offset,
    int32 scales_offset,
    int32 heads,
    int32 query_rows,
    int32 n,
    int32 key_begin) {
    return hmx_prepare_execute_qk_i8_raw_impl(
        handle, buffer_fd, raw_query_offset, raw_key_offset,
        packed_query_offset, packed_key_offset, key_sums_offset,
        output_offset, heads, query_rows, n, key_begin, scales_offset, 1,
        0.0f, 0.0f, 0.0f);
}

/* Select the kth-largest signed INT32 value without materializing a sorted
 * row. Four byte-radix passes identify the exact threshold. A final ascending
 * scan emits indices and resolves threshold ties by the smaller index, which
 * matches the stable CPU Top-k contract. The compact output may alias the
 * score matrix: the write cursor never advances beyond the score currently
 * being consumed. */
static int hmx_i32_topk_rows_in_place(
    int32_t *scores,
    int32_t *indices,
    int rows,
    int row_stride,
    int key_len,
    int causal_prefix_tokens,
    const int32_t *row_offsets) {
    int output_count = 0;
    for (int row = 0; row < rows; ++row) {
        int valid = causal_prefix_tokens + row + 1;
        if (valid > key_len) valid = key_len;
        if (valid < 0) return -1;
        const int keep = row_offsets[row + 1] - row_offsets[row];
        if (keep < 0 || keep > valid) return -1;
        if (keep == 0) continue;

        const int32_t *score_row = scores + (size_t)row * row_stride;
        uint32_t prefix = 0;
        uint32_t prefix_mask = 0;
        int rank = keep;
        for (int pass = 3; pass >= 0; --pass) {
            uint16_t histogram[256] = {0};
            const unsigned shift = (unsigned)pass * 8u;
            for (int column = 0; column < valid; ++column) {
                const uint32_t ordered =
                    ((uint32_t)score_row[column]) ^ 0x80000000u;
                if ((ordered & prefix_mask) == prefix) {
                    ++histogram[(ordered >> shift) & 0xffu];
                }
            }
            int selected_byte = -1;
            for (int byte = 255; byte >= 0; --byte) {
                if (rank > (int)histogram[byte]) {
                    rank -= (int)histogram[byte];
                } else {
                    selected_byte = byte;
                    break;
                }
            }
            if (selected_byte < 0) return -1;
            const uint32_t byte_mask = 0xffu << shift;
            prefix = (prefix & ~byte_mask)
                | ((uint32_t)selected_byte << shift);
            prefix_mask |= byte_mask;
        }

        const int32_t threshold = (int32_t)(prefix ^ 0x80000000u);
        int threshold_remaining = rank;
        const int row_output_begin = output_count;
        for (int column = 0; column < valid; ++column) {
            const int32_t value = score_row[column];
            if (value > threshold
                || (value == threshold && threshold_remaining-- > 0)) {
                indices[output_count++] = column;
            }
        }
        if (output_count - row_output_begin != keep) return -1;
    }
    return output_count;
}

AEEResult hmx_int8_rpc_execute_qk_i32_topk_raw_query(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 raw_query_offset,
    int32 packed_query_offset,
    int32 packed_key_offset,
    int32 key_sums_offset,
    int32 output_offset,
    int32 row_offsets_offset,
    int32 query_rows,
    int32 key_len,
    int32 n,
    int32 causal_prefix_tokens,
    float inverse_q_scale) {
    (void)handle;
    if (buffer_fd < 0 || raw_query_offset < 0 || packed_query_offset < 0 ||
        packed_key_offset < 0 || key_sums_offset < 0 || output_offset < 0 ||
        row_offsets_offset < 0 || query_rows <= 0 ||
        query_rows > HMX_FIXED_M || key_len <= 0 || key_len > n ||
        n <= 0 || n > HMX_FIXED_N_PADDED || (n % 32) != 0 ||
        causal_prefix_tokens < 0 || !(inverse_q_scale > 0.0f) ||
        inverse_q_scale > FLT_MAX) {
        return AEE_EBADPARM;
    }
    uint8_t *buffer = NULL;
    if (HAP_mmap_get(buffer_fd, (void **)&buffer, NULL) || !buffer) {
        return AEE_EFAILED;
    }
    const size_t raw_query_stride = (size_t)HMX_FIXED_M_PADDED
        * HMX_FIXED_K_PADDED * sizeof(float);
    const size_t query_stride =
        (size_t)HMX_FIXED_M_PADDED * HMX_FIXED_K_PADDED;
    const size_t key_stride = (size_t)n * HMX_FIXED_K_PADDED;
    const size_t sums_stride = (size_t)n * sizeof(int32_t);
    float *raw_query = (float *)(buffer + raw_query_offset);
    uint8_t *packed_query = buffer + packed_query_offset;
    int8_t *packed_key = (int8_t *)(buffer + packed_key_offset);
    int32_t *key_sums = (int32_t *)(buffer + key_sums_offset);
    int32_t *output = (int32_t *)(buffer + output_offset);
    int32_t *row_offsets = (int32_t *)(buffer + row_offsets_offset);
    qurt_mem_cache_clean(
        (qurt_addr_t)raw_query, raw_query_stride,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    qurt_mem_cache_clean(
        (qurt_addr_t)packed_key, key_stride,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    qurt_mem_cache_clean(
        (qurt_addr_t)key_sums, sums_stride,
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    qurt_mem_cache_clean(
        (qurt_addr_t)row_offsets,
        (size_t)(query_rows + 1) * sizeof(int32_t),
        QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);

    int status = hmx_i8_hvx_pack_query_f32_ah(
        packed_query, raw_query, query_rows, HMX_FIXED_M_PADDED,
        HMX_FIXED_K_PADDED, inverse_q_scale);
    if (status == 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)packed_query, query_stride,
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }

    hmx_runtime_session temporary_session;
    hmx_runtime_session *session = &s_packed_session;
    int temporary = 0;
    if (status == 0 && !s_packed_session_active) {
        const uint32_t vtcm_bytes = hmx_i8_qk_packed_vtcm_bytes(
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n);
        if (hmx_runtime_begin(&temporary_session, vtcm_bytes) != 0) {
            status = -1;
        } else {
            session = &temporary_session;
            temporary = 1;
        }
    }
    if (status == 0) {
        status = hmx_i8_kernel_qk_i32_packed(
            output, packed_query, packed_key, key_sums,
            HMX_FIXED_M_PADDED, HMX_FIXED_K_PADDED, n,
            session->vtcm_base, session->vtcm_size);
    }
    if (temporary) hmx_runtime_end(&temporary_session);
    int index_count = -1;
    if (status == 0) {
        index_count = hmx_i32_topk_rows_in_place(
            output, output, query_rows, n, key_len,
            causal_prefix_tokens, row_offsets);
        status = index_count < 0 ? -1 : 0;
    }
    if (status == 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)output,
            (size_t)index_count * sizeof(int32_t),
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    HAP_mmap_put(buffer_fd);
    return status == 0 ? AEE_SUCCESS : AEE_EFAILED;
}

AEEResult hmx_int8_rpc_profile_qk_raw_scales(
    remote_handle64 handle,
    int32 buffer_fd,
    int32 raw_query_offset,
    int32 raw_key_offset,
    int32 scales_offset,
    int32 heads,
    int32 query_rows,
    int32 n,
    int32 profile_query,
    int32 profile_key) {
    (void)handle;
    if (buffer_fd < 0 || raw_query_offset < 0 || raw_key_offset < 0 ||
        scales_offset < 0 || heads <= 0 || heads > HMX_FIXED_HEADS ||
        query_rows <= 0 || query_rows > HMX_FIXED_M || n <= 0 ||
        n > HMX_FIXED_N_PADDED || (n % 32) != 0 ||
        (!profile_query && !profile_key)) {
        return AEE_EBADPARM;
    }
    const int incremental_key = profile_key > 1;
    const int key_begin = incremental_key ? profile_key - 2 : 0;
    if (key_begin < 0 || key_begin >= n) return AEE_EBADPARM;
    uint8_t *buffer = NULL;
    int temporary_mapping = 0;
    if (acquire_arena(buffer_fd, &buffer, &temporary_mapping) != 0) {
        return AEE_EFAILED;
    }
    const size_t query_stride = (size_t)HMX_FIXED_M_PADDED
        * HMX_FIXED_K_PADDED * sizeof(float);
    const size_t key_stride = (size_t)n * HMX_FIXED_K_PADDED
        * sizeof(uint16_t);
    float *raw_query = (float *)(buffer + raw_query_offset);
    uint16_t *raw_key = (uint16_t *)(buffer + raw_key_offset);
    float *scales = (float *)(buffer + scales_offset);
    if (profile_query) {
        qurt_mem_cache_clean(
            (qurt_addr_t)raw_query, query_stride * (size_t)heads,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    if (profile_key && !incremental_key) {
        qurt_mem_cache_clean(
            (qurt_addr_t)raw_key, key_stride * (size_t)heads,
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    if (incremental_key) {
        qurt_mem_cache_clean(
            (qurt_addr_t)(scales + heads),
            (size_t)heads * sizeof(float),
            QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    }
    hmx_scale_job job;
    memset(&job, 0, sizeof(job));
    job.raw_query = raw_query;
    job.raw_key = raw_key;
    job.scales = scales;
    job.query_stride = query_stride;
    job.key_stride = key_stride;
    job.heads = heads;
    job.query_rows = query_rows;
    job.n = n;
    job.key_begin = key_begin;
    job.profile_query = profile_query;
    job.profile_key = profile_key != 0;
    job.incremental_key = incremental_key;
    int status = hmx_hvx_parallel_run(hmx_scale_worker, &job);
    if (status == 0) status = job.status;
    if (status == 0) {
        qurt_mem_cache_clean(
            (qurt_addr_t)scales, (size_t)heads * 2u * sizeof(float),
            QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    }
    release_arena(buffer_fd, temporary_mapping);
    return status == 0 ? AEE_SUCCESS : AEE_EFAILED;
}
