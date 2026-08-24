#include "dsp/hvx_worker_pool.h"

#include <HAP_farf.h>
#include <qurt.h>
#include <stdlib.h>
#include <string.h>

enum { HMX_HVX_WORKER_STACK_BYTES = 32 * 1024 };

#ifndef HMX_HVX_COMPILED_LANES
#define HMX_HVX_COMPILED_LANES HMX_HVX_MAX_LANES
#endif

typedef struct hmx_hvx_worker {
    qurt_thread_t thread;
    qurt_sem_t start;
    void *stack;
    int index;
} hmx_hvx_worker;

typedef struct hmx_hvx_pool {
    hmx_hvx_worker workers[HMX_HVX_MAX_WORKERS];
    qurt_sem_t done;
    qurt_mutex_t dispatch_mutex;
    hmx_hvx_worker_callback callback;
    void *context;
    int lanes;
    int active_workers;
    int initialized;
    int stopping;
    int dispatch_active;
} hmx_hvx_pool;

static hmx_hvx_pool s_pool;

static void hmx_hvx_worker_main(void *opaque) {
    hmx_hvx_worker *worker = (hmx_hvx_worker *)opaque;
    for (;;) {
        (void)qurt_sem_down(&worker->start);
        __asm__ __volatile__("barrier" : : : "memory");
        if (s_pool.stopping) break;
        hmx_hvx_worker_callback callback = s_pool.callback;
        void *context = s_pool.context;
        const int lanes = s_pool.lanes;
        if (callback != NULL && worker->index < s_pool.active_workers) {
            callback(context, worker->index + 1, lanes);
            __asm__ __volatile__("barrier" : : : "memory");
            (void)qurt_sem_up(&s_pool.done);
        }
    }
    qurt_thread_exit(0);
}

int hmx_hvx_worker_pool_init(void) {
    if (s_pool.initialized) return 0;
    memset(&s_pool, 0, sizeof(s_pool));

    qurt_sysenv_max_hthreads_t hardware_threads;
    memset(&hardware_threads, 0, sizeof(hardware_threads));
    int maximum_lanes = HMX_HVX_COMPILED_LANES;
    if (qurt_sysenv_get_max_hw_threads(&hardware_threads) == QURT_EOK &&
        hardware_threads.max_hthreads > 0 &&
        hardware_threads.max_hthreads < (unsigned int)maximum_lanes) {
        maximum_lanes = (int)hardware_threads.max_hthreads;
    }
    const unsigned int hvx128 = (qurt_hvx_get_units() >> 8) & 0xffu;
    if (hvx128 > 0u && (int)hvx128 < maximum_lanes) {
        maximum_lanes = (int)hvx128;
    }
    if (maximum_lanes < 1) maximum_lanes = 1;
    s_pool.lanes = maximum_lanes;
    qurt_sem_init_val(&s_pool.done, 0);
    qurt_mutex_init(&s_pool.dispatch_mutex);

    const int priority = qurt_thread_get_priority(qurt_thread_get_id());
    for (int index = 0; index < maximum_lanes - 1; ++index) {
        hmx_hvx_worker *worker = &s_pool.workers[index];
        worker->index = index;
        worker->stack = malloc(HMX_HVX_WORKER_STACK_BYTES);
        if (worker->stack == NULL) {
            s_pool.lanes = index + 1;
            break;
        }
        qurt_sem_init_val(&worker->start, 0);
        qurt_thread_attr_t attributes;
        qurt_thread_attr_init(&attributes);
        qurt_thread_attr_set_stack_addr(&attributes, worker->stack);
        qurt_thread_attr_set_stack_size(
            &attributes, HMX_HVX_WORKER_STACK_BYTES);
        qurt_thread_attr_set_priority(
            &attributes, priority > 0 && priority < 255 ? priority : 64);
        char name[16] = "hmx_hvx_worker0";
        name[14] = (char)('0' + index);
        qurt_thread_attr_set_name(&attributes, name);
        if (qurt_thread_create(
                &worker->thread, &attributes, hmx_hvx_worker_main,
                worker) != QURT_EOK) {
            qurt_sem_destroy(&worker->start);
            free(worker->stack);
            memset(worker, 0, sizeof(*worker));
            s_pool.lanes = index + 1;
            break;
        }
    }
    s_pool.initialized = 1;
    FARF(ALWAYS,
         "hmx_int8: persistent HVX pool lanes=%d hw_threads=%u hvx128=%u",
         s_pool.lanes,
         (unsigned int)(maximum_lanes > 0 ? hardware_threads.max_hthreads : 0),
         hvx128);
    return 0;
}

void hmx_hvx_worker_pool_deinit(void) {
    if (!s_pool.initialized) return;
    qurt_mutex_lock(&s_pool.dispatch_mutex);
    s_pool.stopping = 1;
    __asm__ __volatile__("barrier" : : : "memory");
    for (int index = 0; index < s_pool.lanes - 1; ++index) {
        (void)qurt_sem_up(&s_pool.workers[index].start);
    }
    qurt_mutex_unlock(&s_pool.dispatch_mutex);
    for (int index = 0; index < s_pool.lanes - 1; ++index) {
        hmx_hvx_worker *worker = &s_pool.workers[index];
        if (worker->thread != 0) {
            int status = 0;
            (void)qurt_thread_join(worker->thread, &status);
        }
        qurt_sem_destroy(&worker->start);
        free(worker->stack);
    }
    qurt_sem_destroy(&s_pool.done);
    qurt_mutex_destroy(&s_pool.dispatch_mutex);
    memset(&s_pool, 0, sizeof(s_pool));
}

int hmx_hvx_worker_pool_lanes(void) {
    return s_pool.initialized ? s_pool.lanes : 1;
}

int hmx_hvx_workers_start(hmx_hvx_worker_callback callback, void *context) {
    if (callback == NULL) return -1;
    if (!s_pool.initialized && hmx_hvx_worker_pool_init() != 0) return -1;
    qurt_mutex_lock(&s_pool.dispatch_mutex);
    if (s_pool.dispatch_active) {
        qurt_mutex_unlock(&s_pool.dispatch_mutex);
        return -1;
    }
    s_pool.callback = callback;
    s_pool.context = context;
    s_pool.active_workers = s_pool.lanes - 1;
    s_pool.dispatch_active = 1;
    __asm__ __volatile__("barrier" : : : "memory");
    for (int index = 0; index < s_pool.active_workers; ++index) {
        (void)qurt_sem_up(&s_pool.workers[index].start);
    }
    return 0;
}

int hmx_hvx_workers_wait(void) {
    if (!s_pool.initialized || !s_pool.dispatch_active) return -1;
    for (int index = 0; index < s_pool.active_workers; ++index) {
        (void)qurt_sem_down(&s_pool.done);
    }
    s_pool.callback = NULL;
    s_pool.context = NULL;
    s_pool.active_workers = 0;
    s_pool.dispatch_active = 0;
    qurt_mutex_unlock(&s_pool.dispatch_mutex);
    return 0;
}

int hmx_hvx_parallel_run(hmx_hvx_worker_callback callback, void *context) {
    if (callback == NULL) return -1;
    if (!s_pool.initialized && hmx_hvx_worker_pool_init() != 0) return -1;
    if (s_pool.lanes == 1) {
        callback(context, 0, 1);
        return 0;
    }
    if (hmx_hvx_workers_start(callback, context) != 0) return -1;
    callback(context, 0, s_pool.lanes);
    return hmx_hvx_workers_wait();
}
