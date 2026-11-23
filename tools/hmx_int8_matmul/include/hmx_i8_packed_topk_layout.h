#ifndef HMX_I8_PACKED_TOPK_LAYOUT_H
#define HMX_I8_PACKED_TOPK_LAYOUT_H

#include <stddef.h>
#include <stdint.h>

/* The CPU-packed shared-key RPC reads K from its packed arena. Each
 * unused raw-K slab holds a selection payload followed by the row offsets.
 * Keep the offsets on their own cache lines, away from the DSP output. */
static inline size_t hmx_i8_topk_offsets_start(size_t slab_bytes, int rows) {
    return slab_bytes - (((size_t)(rows + 1) * sizeof(int32_t) + 127) & ~(size_t)127);
}

enum { HMX_I8_TOPK_BITMAP_ROWS = 4, HMX_I8_TOPK_BITMAP_COLS = 32 };

/* Four uint32 masks per tile, with groups truncated at the causal boundary. */
static inline size_t hmx_i8_topk_bitmap_group_words(int first_visible) {
    return (size_t)((first_visible + HMX_I8_TOPK_BITMAP_ROWS - 1
                     + HMX_I8_TOPK_BITMAP_COLS - 1) / HMX_I8_TOPK_BITMAP_COLS)
        * HMX_I8_TOPK_BITMAP_ROWS;
}

static inline size_t hmx_i8_topk_bitmap_words(int rows, int valid_columns) {
    if (rows <= 0 || rows % HMX_I8_TOPK_BITMAP_ROWS != 0
        || valid_columns < rows || valid_columns > UINT16_MAX) {
        return 0;
    }
    size_t words = 0;
    for (int row = 0; row < rows; row += HMX_I8_TOPK_BITMAP_ROWS) {
        words += hmx_i8_topk_bitmap_group_words(valid_columns - rows + row + 1);
    }
    return words;
}

#endif
