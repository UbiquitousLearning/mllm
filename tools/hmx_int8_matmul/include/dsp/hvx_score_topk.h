#ifndef HMX_I8_HVX_SCORE_TOPK_H
#define HMX_I8_HVX_SCORE_TOPK_H

#include <stddef.h>
#include <stdint.h>

/* Four query rows per 128-byte tile, in 32-column order. Offsets describe
 * the requested output count for each causal query row. */
int hmx_i8_hvx_topk_native(const uint8_t *scores, int rows, int padded_columns,
                           int valid_columns, const int32_t *offsets,
                           uint16_t *indices, size_t capacity);

int hmx_i8_hvx_topk_bitmap(const uint8_t *scores, int rows, int padded_columns,
                          int valid_columns, const int32_t *offsets,
                          uint32_t *bitmap, size_t capacity_words);

#endif
