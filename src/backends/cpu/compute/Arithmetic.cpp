#include "Arithmetic.hpp"

void mllm_add_fp32(float *a, float *b, float *c, int n) {
    int i = 0;
    // 使用AVX/AVX2寄存器进行8个浮点数的批量处理
#if defined(__ARM_NEON)
    // 使用NEON寄存器进行4个浮点数的批量处理
    for (i = 0; i <= n - 4; i += 4) {
        // 加载向量
        float32x4_t vec_a = vld1q_f32(&a[i]);
        float32x4_t vec_b = vld1q_f32(&b[i]);

        // 向量加法
        float32x4_t vec_c = vaddq_f32(vec_a, vec_b);

        // 存储结果
        vst1q_f32(&c[i], vec_c);
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (i = 0; i <= n - 8; i += 8) {
        // 加载向量
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);

        // 向量加法
        __m256 vec_c = _mm256_add_ps(vec_a, vec_b);

        // 存储结果
        _mm256_storeu_ps(&c[i], vec_c);
    }
#endif
    // 处理剩余元素
    for (; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}
void mllm_sub_fp32(float *a, float *b, float *c, int n) {
    int i = 0;

    // 使用NEON寄存器进行4个浮点数的批量处理
#if defined(__ARM_NEON)
    for (i = 0; i <= n - 4; i += 4) {
        // 加载向量
        float32x4_t vec_a = vld1q_f32(&a[i]);
        float32x4_t vec_b = vld1q_f32(&b[i]);

        // 向量减法
        float32x4_t vec_c = vsubq_f32(vec_a, vec_b);

        // 存储结果
        vst1q_f32(&c[i], vec_c);
    }
#elif defined(__AVX2__) || defined(__AVX__)
    // 使用AVX/AVX2寄存器进行8个浮点数的批量处理
    for (i = 0; i <= n - 8; i += 8) {
        // 加载向量
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);

        // 向量减法
        __m256 vec_c = _mm256_sub_ps(vec_a, vec_b);

        // 存储结果
        _mm256_storeu_ps(&c[i], vec_c);
    }
#endif
    // 处理剩余元素
    for (; i < n; i++) {
        c[i] = a[i] - b[i];
    }
}
void mllm_mul_fp32(float *a, float *b, float *c, int n) {
    int i = 0;

    // 使用NEON寄存器进行4个浮点数的批量处理
#if defined(__ARM_NEON)
    for (i = 0; i <= n - 4; i += 4) {
        // 加载向量
        float32x4_t vec_a = vld1q_f32(&a[i]);
        float32x4_t vec_b = vld1q_f32(&b[i]);

        // 向量乘法
        float32x4_t vec_c = vmulq_f32(vec_a, vec_b);

        // 存储结果
        vst1q_f32(&c[i], vec_c);
    }
#elif defined(__AVX2__) || defined(__AVX__)
    // 使用AVX/AVX2寄存器进行8个浮点数的批量处理
    for (i = 0; i <= n - 8; i += 8) {
        // 加载向量
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);

        // 向量乘法
        __m256 vec_c = _mm256_mul_ps(vec_a, vec_b);

        // 存储结果
        _mm256_storeu_ps(&c[i], vec_c);
    }
#endif
    // 处理剩余元素
    for (; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}
void mllm_div_fp32(float *a, float *b, float *c, int n) {
    int i = 0;

    // 使用NEON寄存器进行4个浮点数的批量处理
#if defined(__ARM_NEON)
    for (i = 0; i <= n - 4; i += 4) {
        // 加载向量
        float32x4_t vec_a = vld1q_f32(&a[i]);
        float32x4_t vec_b = vld1q_f32(&b[i]);

        // 向量除法
        float32x4_t vec_c = vdivq_f32(vec_a, vec_b);

        // 存储结果
        vst1q_f32(&c[i], vec_c);
    }
#elif defined(__AVX2__) || defined(__AVX__)
    // 使用AVX/AVX2寄存器进行8个浮点数的批量处理
    for (i = 0; i <= n - 8; i += 8) {
        // 加载向量
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_b = _mm256_loadu_ps(&b[i]);

        // 向量除法
        __m256 vec_c = _mm256_div_ps(vec_a, vec_b);

        // 存储结果
        _mm256_storeu_ps(&c[i], vec_c);
    }
#endif
    // 处理剩余元素
    for (; i < n; i++) {
        c[i] = a[i] / b[i];
    }
}

void mllm_add_fp32(float *a, float value, float *c, int n) {
    int i = 0;

    // 使用NEON寄存器进行4个浮点数的批量处理
#if defined(__ARM_NEON)
    // 将标量value扩展为NEON寄存器
    float32x4_t vec_value = vdupq_n_f32(value);

    // 使用NEON进行批量计算
    for (i = 0; i <= n - 4; i += 4) {
        // 加载向量
        float32x4_t vec_a = vld1q_f32(&a[i]);

        // 向量加法: a[i] + value
        float32x4_t vec_c = vaddq_f32(vec_a, vec_value);

        // 存储结果
        vst1q_f32(&c[i], vec_c);
    }

#elif defined(__AVX2__) || defined(__AVX__)
    // 将标量value扩展为AVX寄存器
    __m256 vec_value = _mm256_set1_ps(value);

    // 使用AVX/AVX2进行批量计算
    for (i = 0; i <= n - 8; i += 8) {
        // 加载向量
        __m256 vec_a = _mm256_loadu_ps(&a[i]);

        // 向量加法: a[i] + value
        __m256 vec_c = _mm256_add_ps(vec_a, vec_value);

        // 存储结果
        _mm256_storeu_ps(&c[i], vec_c);
    }
#endif

    // 处理剩余元素
    for (; i < n; i++) {
        c[i] = a[i] + value;
    }
}
void mllm_sub_fp32(float *a, float value, float *c, int n) {
    int i = 0;

#if defined(__ARM_NEON)
    // 将标量value扩展为NEON寄存器
    float32x4_t vec_value = vdupq_n_f32(value);

    // 使用NEON进行批量计算
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t vec_a = vld1q_f32(&a[i]);
        float32x4_t vec_c = vsubq_f32(vec_a, vec_value); // 向量减法: a[i] - value
        vst1q_f32(&c[i], vec_c);
    }

#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vec_value = _mm256_set1_ps(value);

    for (i = 0; i <= n - 8; i += 8) {
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_c = _mm256_sub_ps(vec_a, vec_value); // 向量减法: a[i] - value
        _mm256_storeu_ps(&c[i], vec_c);
    }
#endif

    for (; i < n; i++) {
        c[i] = a[i] - value;
    }
}
void mllm_mul_fp32(float *a, float value, float *c, int n) {
    int i = 0;

#if defined(__ARM_NEON)
    float32x4_t vec_value = vdupq_n_f32(value);

    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t vec_a = vld1q_f32(&a[i]);
        float32x4_t vec_c = vmulq_f32(vec_a, vec_value); // 向量乘法: a[i] * value
        vst1q_f32(&c[i], vec_c);
    }

#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vec_value = _mm256_set1_ps(value);

    for (i = 0; i <= n - 8; i += 8) {
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_c = _mm256_mul_ps(vec_a, vec_value); // 向量乘法: a[i] * value
        _mm256_storeu_ps(&c[i], vec_c);
    }
#endif

    for (; i < n; i++) {
        c[i] = a[i] * value;
    }
}
void mllm_div_fp32(float *a, float value, float *c, int n) {
    int i = 0;

#if defined(__ARM_NEON)
    float32x4_t vec_value = vdupq_n_f32(value);

    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t vec_a = vld1q_f32(&a[i]);
        float32x4_t vec_c = vdivq_f32(vec_a, vec_value); // 向量除法: a[i] / value
        vst1q_f32(&c[i], vec_c);
    }

#elif defined(__AVX2__) || defined(__AVX__)
    __m256 vec_value = _mm256_set1_ps(value);

    for (i = 0; i <= n - 8; i += 8) {
        __m256 vec_a = _mm256_loadu_ps(&a[i]);
        __m256 vec_c = _mm256_div_ps(vec_a, vec_value); // 向量除法: a[i] / value
        _mm256_storeu_ps(&c[i], vec_c);
    }
#endif

    for (; i < n; i++) {
        c[i] = a[i] / value;
    }
}
