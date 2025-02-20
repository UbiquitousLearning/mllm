
// 平台检测头文件
#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// SIMD加速的内存复制函数
inline void simd_memcpy(float* dst, const float* src, size_t n) {
    size_t i = 0;
#if defined(__AVX__)
    // AVX版本处理8个float一组
    for (; i + 7 < n; i += 8) {
        __m256 chunk = _mm256_loadu_ps(src + i);
        _mm256_storeu_ps(dst + i, chunk);
    }
#elif defined(__ARM_NEON)
    // NEON版本处理4个float一组
    for (; i + 3 < n; i += 4) {
        float32x4_t chunk = vld1q_f32(src + i);
        vst1q_f32(dst + i, chunk);
    }
#endif
    // 处理剩余元素
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
}
