#ifndef HMX_I8_TOPK_BITMAP_DECODE_H
#define HMX_I8_TOPK_BITMAP_DECODE_H

#include "hmx_i8_packed_topk_layout.h"

/* A four-row group is the minimum work unit. Readers of disjoint row ranges
 * can decode the same payload concurrently into disjoint index ranges. */
static inline int hmx_i8_decode_topk_bitmap_rows(
    const void *bitmap, size_t bitmap_bytes, int rows, int valid_columns,
    int row_begin, int row_end, const int32_t *offsets,
    int32_t *indices, size_t index_capacity) {
    const size_t words = hmx_i8_topk_bitmap_words(rows, valid_columns);
    if (bitmap == NULL || (uintptr_t)bitmap % sizeof(uint32_t) != 0
        || words == 0 || words > bitmap_bytes / sizeof(uint32_t)
        || offsets == NULL || indices == NULL || offsets[0] != 0
        || offsets[rows] <= 0 || (size_t)offsets[rows] > index_capacity
        || row_begin < 0 || row_end < row_begin || row_end > rows
        || row_begin % HMX_I8_TOPK_BITMAP_ROWS != 0 || row_end % HMX_I8_TOPK_BITMAP_ROWS != 0) {
        return -1;
    }

    const uint32_t *group = (const uint32_t *)bitmap;
    for (int row = 0; row < row_begin; row += HMX_I8_TOPK_BITMAP_ROWS) {
        group += hmx_i8_topk_bitmap_group_words(valid_columns - rows + row + 1);
    }
    for (int row = row_begin; row < row_end; row += HMX_I8_TOPK_BITMAP_ROWS) {
        const int first_visible = valid_columns - rows + row + 1;
        const size_t group_words = hmx_i8_topk_bitmap_group_words(first_visible);
        int32_t cursor[HMX_I8_TOPK_BITMAP_ROWS];
        int32_t end[HMX_I8_TOPK_BITMAP_ROWS];
        for (int lane = 0; lane < HMX_I8_TOPK_BITMAP_ROWS; ++lane) {
            cursor[lane] = offsets[row + lane];
            end[lane] = offsets[row + lane + 1];
            if (cursor[lane] < 0 || end[lane] <= cursor[lane] || end[lane] > offsets[rows]
                || end[lane] - cursor[lane] > first_visible + lane) {
                return -1;
            }
        }
        int column = 0;
        for (size_t word = 0; word < group_words; word += HMX_I8_TOPK_BITMAP_ROWS, column += HMX_I8_TOPK_BITMAP_COLS) {
            for (int lane = 0; lane < HMX_I8_TOPK_BITMAP_ROWS; ++lane) {
                uint32_t mask = group[word + lane];
                const int left = first_visible + lane - column;
                if (__builtin_popcount(mask) > end[lane] - cursor[lane]
                    || (left <= 0 && mask != 0)
                    || (left > 0 && left < HMX_I8_TOPK_BITMAP_COLS && (mask >> left) != 0)) {
                    return -1;
                }
                while (mask != 0) {
                    indices[cursor[lane]++] = column + __builtin_ctz(mask);
                    mask &= mask - 1;
                }
            }
        }
        for (int lane = 0; lane < HMX_I8_TOPK_BITMAP_ROWS; ++lane) {
            if (cursor[lane] != end[lane]) return -1;
        }
        group += group_words;
    }
    return 0;
}

#endif
