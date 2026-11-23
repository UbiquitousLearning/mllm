#ifndef HMX_INT8_HVX_WORKER_POOL_H
#define HMX_INT8_HVX_WORKER_POOL_H

/* The FastRPC handler is lane zero.  Three persistent QuRT workers provide
 * four-way HVX execution without constructing threads in an attention call. */
enum { HMX_HVX_MAX_LANES = 4, HMX_HVX_MAX_WORKERS = 3 };

typedef void (*hmx_hvx_worker_callback)(void *context, int lane, int lanes);

int hmx_hvx_worker_pool_init(void);
void hmx_hvx_worker_pool_deinit(void);
int hmx_hvx_worker_pool_lanes(void);

/* Run callback on the FastRPC handler and up to three persistent workers. */
int hmx_hvx_parallel_run(hmx_hvx_worker_callback callback, void *context);

/* Launch only the persistent workers.  The caller may execute HMX while the
 * callbacks prepare disjoint data in DDR, then joins with workers_wait(). */
int hmx_hvx_workers_start(hmx_hvx_worker_callback callback, void *context);
int hmx_hvx_workers_wait(void);

#endif /* HMX_INT8_HVX_WORKER_POOL_H */
