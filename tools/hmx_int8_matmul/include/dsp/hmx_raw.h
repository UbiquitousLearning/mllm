#ifndef HMX_INT8_RAW_H
#define HMX_INT8_RAW_H

#include <stddef.h>
#include <stdint.h>

enum {
    HMX_I8_M_TILE = 64,
    HMX_I8_K_TILE = 32,
    HMX_I8_N_TILE = 32,
    HMX_I8_ACT_TILE_BYTES = 2048,
    HMX_I8_WEIGHT_TILE_BYTES = 1024,
    HMX_I8_OUTPUT_TILE_BYTES = 8192,
};

static inline __attribute__((always_inline)) void hmx_raw_clear_i32(void) {
    /* v75 opcode from the local disassembly: 11 c0 e0 a6 */
    asm volatile("mxclracc" ::: "memory");
}

static inline __attribute__((always_inline)) void hmx_raw_mac_u8i8(
    const uint8_t *activation,
    const int8_t *weight) {
    /*
     * Direct equivalents of the two instructions recovered from
     * hexkl_micro_hmx_mm_u8i8:
     *
     *   ed 43 04 92  activation.ub = mxmem(act, 31):cm
     *   e0 e5 02 92  weight.b      = mxmem(weight, 896)
     *
     * The operands are the final vector indices, not byte counts.
     */
    const int activation_limit = 31;
    const int weight_limit = 896;
    asm volatile(
        "{ activation.ub = mxmem(%0, %1):cm\n"
        "  weight.b = mxmem(%2, %3) }\n"
        :
        : "r"(activation), "r"(activation_limit),
          "r"(weight), "r"(weight_limit)
        : "memory");
}

static inline __attribute__((always_inline)) void hmx_raw_mac_u8i8_deep(
    const uint8_t *activation,
    const int8_t *weight,
    int k_tiles) {
    /*
     * Stream a contiguous run of 64x32 AH activation tiles and 32x32 WH
     * weight tiles through one HMX packet.  These limits are instruction
     * encodings, not byte counts:
     *
     *   activation: ((depth - 1) << 11) | 31
     *   weight:     (depth << 10) - 128
     *
     * The encoding is the one used by the local HTPOPLIB W8A8 kernel.  A
     * single packet supports at most 32 K tiles, so longer reductions are
     * emitted as multiple deep packets while retaining the accumulator.
     */
    const int max_depth = 32;
    for (int tile = 0; tile < k_tiles; tile += max_depth) {
        const int remaining = k_tiles - tile;
        const int depth = remaining < max_depth ? remaining : max_depth;
        const uint32_t activation_limit =
            (uint32_t)(((depth - 1) << 11) | 31);
        const uint32_t weight_limit = (uint32_t)((depth << 10) - 128);
        const uint8_t *activation_ptr =
            activation + (size_t)tile * HMX_I8_ACT_TILE_BYTES;
        const int8_t *weight_ptr =
            weight + (size_t)tile * HMX_I8_WEIGHT_TILE_BYTES;
        asm volatile(
            "{ activation.ub = mxmem(%0, %1):deep:cm\n"
            "  weight.b = mxmem(%2, %3) }\n"
            :
            : "r"(activation_ptr), "r"(activation_limit),
              "r"(weight_ptr), "r"(weight_limit)
            : "memory");
    }
}

static inline __attribute__((always_inline)) void hmx_raw_read_acc_byte_plane(
    uint8_t *output,
    const void *scale_bias) {
    /*
     * Exact readback sequence recovered from the local v75 object:
     *
     *   bias = mxmem(scale_bias)
     *   mxmem(output, 0):after:retain:cm.ub = acc
     *
     * retain is required because four differently biased reads recover the
     * four bytes of every INT32 accumulator lane.
     */
    register int offset asm("r2") = 0;
    asm volatile(
        "%2 = #0\n\t"
        "bias = mxmem(%1)\n\t"
        "mxmem(%0, %2):after:retain:cm.ub = acc\n"
        :
        : "r"(output), "r"(scale_bias), "r"(offset)
        : "r2", "memory");
}

#endif /* HMX_INT8_RAW_H */
