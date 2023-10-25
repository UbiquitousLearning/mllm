//
// Created by ey on 23-10-23.
//

#include <cstring>
#include "QuantizeQ4.hpp"

// reference implementation for deterministic creation of model files
void quantize_row_q4_0_reference(const float *x, block_q4_0 *y, int k) {
    static const int Qk = QK4_0;

    assert(k % Qk == 0);

    const int nb = k / Qk;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max = 0.0f;

        for (int j = 0; j < Qk; j++) {
            const float v = x[i * Qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v);
                max = v;
            }
        }

        const float d = max / -8;
        const float id = d ? 1.0f / d : 0.0f;

        //        y[i].d = _cvtss_sh(d, 0);
        y[i].d = d;

        for (int j = 0; j < Qk / 2; ++j) {
            const float x0 = x[i * Qk + 0 + j] * id;
            const float x1 = x[i * Qk + Qk / 2 + j] * id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j] = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}

// static float table_f32_f16[1 << 16];
//
// inline static float mllm_lookup_fp16_to_fp32(mllm_fp16_t f) {
//     uint16_t s;
//     memcpy(&s, &f, sizeof(uint16_t));
//     return table_f32_f16[s];
// }

void dequantize_row_q4_0(const block_q4_0 *x, float *y, int k) {
    static const int Qk = QK4_0;

    assert(k % Qk == 0);

    const int nb = k / Qk;

    for (int i = 0; i < nb; i++) {
        //        const float d = mllm_lookup_fp16_to_fp32(x[i].d);
        const float d = x[i].d;

        for (int j = 0; j < Qk / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;

            y[i * Qk + j + 0] = x0 * d;
            y[i * Qk + j + Qk / 2] = x1 * d;
        }
    }
}
