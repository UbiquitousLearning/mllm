#ifdef __ARM_NEON
#define FP16_TO_FP32(x) ((float)(x))
#define FP32_TO_FP16(x) (x)
#define F32_VEC float32x4_t
#define F32_STEP 16                // 16 elements per step
#define F32_REG 4                  // 4 elements per register
#define F32_ARR F32_STEP / F32_REG // Len of sum array
#define F32_VEC_REDUCE(res, x)                     \
    {                                              \
        int offset = F32_ARR >> 1;                 \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        offset >>= 1;                              \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        offset >>= 1;                              \
        for (int i = 0; i < offset; ++i) {         \
            x[i] = vaddq_f32(x[i], x[offset + i]); \
        }                                          \
        res = vaddvq_f32(x[0]);                    \
    }
#endif