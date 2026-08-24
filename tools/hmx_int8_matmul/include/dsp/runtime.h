#ifndef HMX_INT8_RUNTIME_H
#define HMX_INT8_RUNTIME_H

#include <stdint.h>

int hmx_runtime_setup(void);
void hmx_runtime_reset(void);

typedef struct hmx_runtime_session {
    int resource_context;
    uint8_t *vtcm_base;
    uint32_t vtcm_size;
    int hmx_locked;
} hmx_runtime_session;

int hmx_runtime_begin(hmx_runtime_session *session, uint32_t minimum_vtcm_size);
void hmx_runtime_end(hmx_runtime_session *session);

#endif /* HMX_INT8_RUNTIME_H */
