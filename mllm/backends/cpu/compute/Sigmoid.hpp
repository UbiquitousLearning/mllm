#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h> // 包含 NEON 指令集的头文件
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h> // 包含 AVX, SSE 等指令集的头文件
#endif
#include <cmath>

#if defined(__AVX2__) && defined(__FMA__)
// AVX2 版本的快速 exp (示意)
static inline __m256 fast_exp_ps_avx2(__m256 x) {
    float temp_in[8], temp_out[8];
    _mm256_storeu_ps(temp_in, x);
    for (int i = 0; i < 8; ++i) temp_out[i] = expf(temp_in[i]);
    return _mm256_loadu_ps(temp_out);
}
#endif

#if defined(__ARM_NEON) && defined(__aarch64__)
static inline float32x4_t fast_exp_f32_neon(float32x4_t x) {
    float temp_in[4], temp_out[4];
    vst1q_f32(temp_in, x);
    for (int i = 0; i < 4; ++i) temp_out[i] = expf(temp_in[i]);
    return vld1q_f32(temp_out);
}
#endif

/**
 * @brief 对一个 float 数组进行 Sigmoid 计算 (支持 AVX 和 NEON 的高性能版本)
 * @param n   数组中元素的数量
 * @param y   指向输出数组的指针
 * @param x   指向输入数组的指针
 */
void vec_sigmoid_f32(const int n, float *y, const float *x) {
    int i = 0;

// 1. 优先使用 AVX2 和 FMA 指令集 (x86 架构, 一次处理8个float)
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 ones_avx = _mm256_set1_ps(1.0f);
    const __m256 zeros_avx = _mm256_setzero_ps();

    for (; i + 7 < n; i += 8) {
        __m256 val = _mm256_loadu_ps(x + i); // 加载数据
        val = _mm256_sub_ps(zeros_avx, val); // 计算 -x
        val = fast_exp_ps_avx2(val);         // 计算 exp(-x)
        val = _mm256_add_ps(ones_avx, val);  // 计算 1 + exp(-x)
        val = _mm256_div_ps(ones_avx, val);  // 计算 1 / (...)
        _mm256_storeu_ps(y + i, val);        // 存储结果
    }

// 2. 其次，如果平台是 ARMv8-A (aarch64)，则使用 NEON (一次处理4个float)
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const float32x4_t ones_neon = vdupq_n_f32(1.0f);

    for (; i + 3 < n; i += 4) {
        float32x4_t val = vld1q_f32(x + i); // 加载数据
        val = vnegq_f32(val);               // 计算 -x
        val = fast_exp_f32_neon(val);       // 计算 exp(-x)
        val = vaddq_f32(ones_neon, val);    // 计算 1 + exp(-x)
        val = vdivq_f32(ones_neon, val);    // 计算 1 / (...) (vdivq_f32 在 aarch64 中可用)
        vst1q_f32(y + i, val);              // 存储结果
    }
#endif

    // 3. "收尾"循环：处理剩余的不足一个SIMD块的元素，或在不支持SIMD的平台上运行
    for (; i < n; ++i) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}