//
// Created by ey on 23-10-23.
//

#include <cstring>
#include "QuantizeQ4.hpp"


// reference implementation for deterministic creation of model files
void quantize_row_q4_0_reference(const float * __restrict x, block_q4_0  *__restrict y, int k) {
    static const int Qk = QK4_0;

    assert(k % Qk == 0);

    const int nb = k / Qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0F; // absolute max
        float max = 0.0F;

        for (int j = 0; j < Qk; j++) {
            const float v = x[i * Qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0F / d : 0.0F;

        y[i].d = MLLM_FP32_TO_FP16(d);

        for (int j = 0; j < Qk / 2; ++j) {
            const float x0 = x[i * Qk + 0 + j] * id;
            const float x1 = x[i * Qk + Qk / 2 + j] * id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5F));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5F));

            y[i].qs[j] = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

void quantize_row_q4_0(const float * __restrict x, void * __restrict y, int k) {
    quantize_row_q4_0_reference(x, (block_q4_0 *)y, k);
}

void dequantize_row_q4_0(const void * __restrict vx, float * __restrict y, int k) {
    static const int Qk = QK4_0;

    assert(k % Qk == 0);

    block_q4_0 * __restrict x = (block_q4_0 *)vx;
    const int nb = k / Qk;

    for (int i = 0; i < nb; i++) {
        const float d = MLLM_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < Qk / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;

            y[i * Qk + j + 0] = x0 * d;
            y[i * Qk + j + Qk / 2] = x1 * d;
        }
    }
}
