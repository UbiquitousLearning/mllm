#ifndef HMX_INT8_DMA_UTILS_H
#define HMX_INT8_DMA_UTILS_H

#include <hexagon_protos.h>
#include <stdbool.h>
#include <stdint.h>

enum {
    HMX_DMA_STATUS_MASK = 3,
    HMX_DMA_STATUS_IDLE = 0,
    HMX_DMA_DESC_PENDING = 0,
    HMX_DMA_DESC_TYPE_1D = 0,
};

typedef struct hmx_dma_desc_1d {
    uint32_t next;
    union {
        struct {
            unsigned length : 24;
            unsigned type : 2;
            unsigned dst_dlbc : 1;
            unsigned src_dlbc : 1;
            unsigned dst_bypass : 1;
            unsigned src_bypass : 1;
            unsigned ordered : 1;
            unsigned dstate : 1;
        } __attribute__((packed));
        uint32_t control;
    };
    uint32_t src;
    uint32_t dst;
} __attribute__((packed)) hmx_dma_desc_1d;

static inline void hmx_dma_start(void *descriptor) {
    asm volatile("release(%0):at" : : "r"(descriptor));
    Q6_dmstart_A(descriptor);
}

static inline bool hmx_dma_wait_idle(void) {
    return (Q6_R_dmwait() & HMX_DMA_STATUS_MASK) == HMX_DMA_STATUS_IDLE;
}

static inline void hmx_dma_prepare_copy(
    hmx_dma_desc_1d *descriptor,
    const void *source,
    void *destination,
    uint32_t bytes) {
    descriptor->next = 0;
    descriptor->control = 0;
    descriptor->length = bytes;
    descriptor->type = HMX_DMA_DESC_TYPE_1D;
    descriptor->src_bypass = 1;
    descriptor->dst_bypass = 0;
    descriptor->ordered = 1;
    descriptor->dstate = HMX_DMA_DESC_PENDING;
    descriptor->src = (uint32_t)(uintptr_t)source;
    descriptor->dst = (uint32_t)(uintptr_t)destination;
}

#endif
