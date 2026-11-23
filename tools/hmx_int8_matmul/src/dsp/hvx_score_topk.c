#include "dsp/hvx_score_topk.h"
#include "hmx_i8_packed_topk_layout.h"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

enum { group_rows = HMX_I8_TOPK_BITMAP_ROWS, tile_cols = HMX_I8_TOPK_BITMAP_COLS,
       bins = 256, vector_bytes = 128 };
typedef uint16_t Histograms[group_rows][bins];
typedef uint16_t LaneHistograms[32][group_rows][2][8];

extern void hmx_i8_hvx_histogram_tiles(const uint8_t *tiles, unsigned count, LaneHistograms lanes);

static unsigned tile_offset(int lane, int col) {
    return (unsigned)((col / tile_cols) * vector_bytes + lane * tile_cols + col % tile_cols);
}

static void tile_histogram(const uint8_t *scores, int visible, Histograms hist) {
    const int complete = visible / tile_cols;
    if (complete > 0) {
        _Alignas(vector_bytes) LaneHistograms lanes;
        hmx_i8_hvx_histogram_tiles(scores, (unsigned)complete, lanes);
        for (int group = 0; group < 32; ++group)
            for (int lane = 0; lane < group_rows; ++lane)
                for (int bin = 0; bin < 8; ++bin)
                    hist[lane][group * 8 + bin] = lanes[group][lane][0][bin] + lanes[group][lane][1][bin];
    } else {
        memset(hist, 0, sizeof(Histograms));
    }
    for (int lane = 0; lane < group_rows; ++lane)
        for (int col = complete * tile_cols; col < visible + lane; ++col)
            ++hist[lane][scores[tile_offset(lane, col)]];
}

static inline HVX_Vector pack_predicate(HVX_VectorPred predicate) {
    // Each four-byte dot product packs four adjacent predicate bytes into a nibble.
    const HVX_Vector weighted = Q6_V_vand_QR(predicate, 0x08040201);
    const HVX_Vector words = Q6_Vuw_vrmpy_VubRub(weighted, 0x01010101);
    const HVX_Vector halfwords = Q6_Vh_vpacke_VwVw(words, words);
    return Q6_Vb_vpacke_VhVh(halfwords, halfwords);
}

static inline uint32_t dense_nibbles(uint32_t value) {
    value = (value | (value >> 4)) & UINT32_C(0x00ff00ff);
    return (value | (value >> 8)) & UINT32_C(0xffff);
}

static inline uint32_t row_mask(HVX_Vector packed, int lane) {
    const uint32_t low = (uint32_t)Q6_R_vextract_VR(packed, lane * 8);
    const uint32_t high = (uint32_t)Q6_R_vextract_VR(packed, lane * 8 + 4);
    return dense_nibbles(low) | (dense_nibbles(high) << 16);
}

static uint32_t take_equal_prefix(uint32_t equal, int *remaining) {
    const int count = __builtin_popcount(equal);
    if (count <= *remaining) {
        *remaining -= count;
        return equal;
    }
    uint32_t result = 0;
    while (*remaining > 0) {
        const uint32_t lowest = equal & (0u - equal);
        result |= lowest;
        equal ^= lowest;
        --*remaining;
    }
    return result;
}

static inline __attribute__((always_inline)) int emit_four_rows(
    const uint8_t *block, int first_visible, const int thresholds[group_rows],
    int ties[group_rows], const int keep[group_rows], uint16_t *out[group_rows],
    uint32_t *bitmap, int emit_bitmap) {
    union { HVX_Vector vector; uint8_t bytes[vector_bytes]; } cutoffs;
    for (int lane = 0; lane < group_rows; ++lane)
        memset(cutoffs.bytes + lane * tile_cols, (uint8_t)thresholds[lane], tile_cols);
    int emitted[group_rows] = {0};
    // The final tile may be only partially visible or absent in the first rows.
    for (int column = 0; column < first_visible + group_rows - 1; column += tile_cols) {
        HVX_Vector scores;
        memcpy(&scores, block + (column / tile_cols) * vector_bytes, sizeof(scores));
        const HVX_Vector greater = pack_predicate(Q6_Q_vcmp_gt_VbVb(scores, cutoffs.vector));
        const HVX_Vector equal = pack_predicate(Q6_Q_vcmp_eq_VbVb(scores, cutoffs.vector));
        for (int lane = 0; lane < group_rows; ++lane) {
            const int left = first_visible + lane - column;
            if (left <= 0) {
                if (emit_bitmap) bitmap[lane] = 0;
                continue;
            }
            const uint32_t valid = left >= tile_cols ? UINT32_MAX : UINT32_MAX >> (tile_cols - left);
            uint32_t selected = row_mask(greater, lane) & valid;
            if (ties[lane] > 0) {
                const uint32_t matching = row_mask(equal, lane) & valid;
                selected |= take_equal_prefix(matching, &ties[lane]);
            }
            if (emit_bitmap) {
                emitted[lane] += __builtin_popcount(selected);
                bitmap[lane] = selected;
            } else {
                if (__builtin_popcount(selected) > keep[lane] - emitted[lane]) return -1;
                while (selected != 0) {
                    const int bit = __builtin_ctz(selected);
                    out[lane][emitted[lane]++] = (uint16_t)(column + bit);
                    selected &= selected - 1;
                }
            }
        }
        if (emit_bitmap) bitmap += group_rows;
    }
    for (int lane = 0; lane < group_rows; ++lane)
        if (emitted[lane] != keep[lane] || ties[lane] != 0) return -1;
    return 0;
}

static inline __attribute__((always_inline)) int select_topk(
    const uint8_t *scores, int rows, int padded_columns, int valid_columns,
    const int32_t *offsets, uint16_t *indices, uint32_t *bitmap, size_t capacity,
    int emit_bitmap) {
    if (scores == NULL || offsets == NULL ||
        (emit_bitmap ? bitmap == NULL : indices == NULL) ||
        ((uintptr_t)scores & (vector_bytes - 1)) != 0 ||
        rows <= 0 || rows % group_rows != 0 ||
        valid_columns < rows || valid_columns > UINT16_MAX ||
        padded_columns < valid_columns || padded_columns % tile_cols != 0 ||
        offsets[0] != 0 || offsets[rows] <= 0 ||
        (emit_bitmap ? hmx_i8_topk_bitmap_words(rows, valid_columns) : (size_t)offsets[rows]) > capacity) {
        return -1;
    }
    for (int row = 0; row < rows; ++row) {
        const int previous = offsets[row];
        const int next = offsets[row + 1];
        if (next <= previous || next > offsets[rows] ||
            next - previous > valid_columns - rows + row + 1) {
            return -1;
        }
    }

    for (int group = 0; group < rows; group += group_rows) {
        const int visible = valid_columns - rows + group + 1;
        const uint8_t *block = scores + (size_t)group * padded_columns;
        Histograms histogram;
        int thresholds[group_rows], ties[group_rows], keep[group_rows];
        uint16_t *out[group_rows];
        tile_histogram(block, visible, histogram);
        for (int lane = 0; lane < group_rows; ++lane) {
            const int row = group + lane;
            keep[lane] = offsets[row + 1] - offsets[row];
            if (!emit_bitmap) out[lane] = indices + offsets[row];
            int remaining = keep[lane];
            int threshold = 127;
            while (threshold >= -128 && remaining > histogram[lane][(uint8_t)threshold]) {
                remaining -= histogram[lane][(uint8_t)threshold];
                --threshold;
            }
            if (threshold < -128) return -1;
            thresholds[lane] = threshold;
            ties[lane] = remaining;
        }
        if (emit_four_rows(block, visible, thresholds, ties, keep, out, bitmap, emit_bitmap) != 0) {
            return -1;
        }
        if (emit_bitmap) bitmap += hmx_i8_topk_bitmap_group_words(visible);
    }
    return 0;
}

int hmx_i8_hvx_topk_native(const uint8_t *scores, int rows, int padded_columns,
                          int valid_columns, const int32_t *offsets,
                          uint16_t *indices, size_t capacity) {
    return select_topk(scores, rows, padded_columns, valid_columns, offsets, indices, NULL, capacity, 0);
}

int hmx_i8_hvx_topk_bitmap(const uint8_t *scores, int rows, int padded_columns,
                          int valid_columns, const int32_t *offsets,
                          uint32_t *bitmap, size_t capacity_words) {
    return select_topk(scores, rows, padded_columns, valid_columns, offsets, NULL, bitmap, capacity_words, 1);
}
