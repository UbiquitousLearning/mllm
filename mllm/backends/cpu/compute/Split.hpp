// #include <iostream>
// #include <vector>
// #include <numeric>
// #include <stdexcept>
// #include <cstring> // for memcpy

// // 引入 OpenMP 头文件
// #include <omp.h>

// // 引入SIMD头文件
// #if defined(__AVX__)
// #include <immintrin.h>
// #elif defined(__ARM_NEON)
// #include <arm_neon.h>
// #endif

// /**
//  * @brief 高效地将一个4D张量按指定维度分割 (OpenMP并行版本)
//  * @param origin 输入的4D张量的裸指针
//  * @param origin_dims 输入张量的维度信息，大小为4的数组，例如 {N, C, H, W}
//  * @param out 输出张量指针的向量
//  * @param split_dims 每个输出张量在分割维度上的大小
//  * @param dim_id 要进行分割的维度索引 (0到3)
//  */
// void efficient_split(const float *origin, const int *origin_dims,
//                      std::vector<float *> &out, const std::vector<int> &split_dims,
//                      int dim_id) {
//     // --- 1. 输入验证 ---
//     if (dim_id < 0 || dim_id > 3) {
//         throw std::invalid_argument("Error: dim_id must be between 0 and 3.");
//     }

//     int total_split_dim = std::accumulate(split_dims.begin(), split_dims.end(), 0);
//     if (total_split_dim != origin_dims[dim_id]) {
//         throw std::invalid_argument("Error: Sum of split_dims must be equal to the dimension size of origin tensor.");
//     }

//     if (out.size() != split_dims.size()) {
//         throw std::invalid_argument("Error: The size of 'out' vector must be equal to the size of 'split_dims' vector.");
//     }

//     // --- 2. 计算步长和循环尺寸 ---
//     int strides[4];
//     strides[3] = 1;
//     for (int i = 2; i >= 0; --i) {
//         strides[i] = strides[i + 1] * origin_dims[i + 1];
//     }

//     int outer_loop_size = 1;
//     for (int i = 0; i < dim_id; ++i) {
//         outer_loop_size *= origin_dims[i];
//     }

//     int inner_loop_size = strides[dim_id];
//     int original_dim_size_at_split_axis = origin_dims[dim_id] * inner_loop_size;

//     // --- 3. 为并行计算预先计算偏移量 ---
//     // 为了使主循环能够并行化，我们需要为每个输出张量预先计算其在源张量中的起始偏移量。
//     // 这避免了在循环中依赖于前一次迭代结果的顺序更新。
//     std::vector<int> split_offsets(split_dims.size() + 1, 0);
//     for (size_t i = 0; i < split_dims.size(); ++i) {
//         split_offsets[i + 1] = split_offsets[i] + split_dims[i];
//     }

//     // --- 4. 并行处理 ---
//     // 使用 OpenMP 对主循环进行并行化。
//     // OpenMP 会自动根据系统配置或 OMP_NUM_THREADS 环境变量来决定使用的线程数。
//     // 循环变量 'i' 默认是私有的。所有在循环外声明的变量都是共享的，
//     // 这在这里是安全的，因为它们在并行区域内是只读的（除了 'out'，但每个线程访问 out[i]，不会冲突）。
//     #pragma omp parallel for
//     for (size_t i = 0; i < out.size(); ++i) {
//         float *out_ptr = out[i];
//         const int split_size = split_dims[i];
//         const int offset_in_dim = split_offsets[i];

//         // 遍历所有不被分割的外部维度
//         for (int outer_idx = 0; outer_idx < outer_loop_size; ++outer_idx) {
//             // 计算源和目标在当前外部维度块的基地址
//             const float *src_base = origin + outer_idx * original_dim_size_at_split_axis + offset_in_dim * inner_loop_size;
//             float *dst_base = out_ptr + outer_idx * split_size * inner_loop_size;

//             // 沿着分割轴，拷贝 'split_size' 个大小为 'inner_loop_size' 的数据块
//             for (int split_idx = 0; split_idx < split_size; ++split_idx) {
//                 const float *src = src_base + split_idx * inner_loop_size;
//                 float *dst = dst_base + split_idx * inner_loop_size;
//                 int count = inner_loop_size;

//                 // 使用SIMD指令集进行高效的内存拷贝
// #if defined(__AVX__)
//                 for (; count >= 8; count -= 8) {
//                     __m256 data = _mm256_loadu_ps(src);
//                     _mm256_storeu_ps(dst, data);
//                     src += 8;
//                     dst += 8;
//                 }
// #elif defined(__ARM_NEON)
//                 for (; count >= 4; count -= 4) {
//                     float32x4_t data = vld1q_f32(src);
//                     vst1q_f32(dst, data);
//                     src += 4;
//                     dst += 4;
//                 }
// #endif
//                 // 处理剩余不足一个SIMD寄存器大小的数据
//                 for (; count > 0; --count) {
//                     *dst++ = *src++;
//                 }
//             }
//         }
//     }
// }

// Split.hpp

#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstring> // for memcpy

// 引入 OpenMP 头文件
#include <omp.h>

// 引入SIMD头文件
#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// 新增：引入项目所需的数据类型和转换定义
#include "Types.hpp"
#include "backends/cpu/third_party/ggml/QuantizeFP16.hpp"

/**
 * @brief 高效地将一个4D张量按指定维度分割 (OpenMP并行版本)，支持混合输出类型(float/fp16)
 * @param origin 输入的4D张量的裸指针 (必须是 float 类型)
 * @param origin_dims 输入张量的维度信息，大小为4的数组，例如 {N, C, H, W}
 * @param out 输出张量裸指针的向量 (void*)
 * @param out_types 每个输出张量的数据类型
 * @param split_dims 每个输出张量在分割维度上的大小
 * @param dim_id 要进行分割的维度索引 (0到3)
 */
void efficient_split(const float *origin, const int *origin_dims,
                     std::vector<void *> &out, const std::vector<DataType> &out_types, // 修改点
                     const std::vector<int> &split_dims,
                     int dim_id) {
    // --- 1. 输入验证 ---
    if (dim_id < 0 || dim_id > 3) {
        throw std::invalid_argument("Error: dim_id must be between 0 and 3.");
    }

    int total_split_dim = std::accumulate(split_dims.begin(), split_dims.end(), 0);
    if (total_split_dim != origin_dims[dim_id]) {
        throw std::invalid_argument("Error: Sum of split_dims must be equal to the dimension size of origin tensor.");
    }

    if (out.size() != split_dims.size() || out.size() != out_types.size()) { // 修改点
        throw std::invalid_argument("Error: The size of 'out', 'out_types', and 'split_dims' vectors must be equal.");
    }

    // --- 2. 计算步长和循环尺寸 ---
    int strides[4];
    strides[3] = 1;
    for (int i = 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * origin_dims[i + 1];
    }

    int outer_loop_size = 1;
    for (int i = 0; i < dim_id; ++i) {
        outer_loop_size *= origin_dims[i];
    }

    int inner_loop_size = strides[dim_id];
    int original_dim_size_at_split_axis = origin_dims[dim_id] * inner_loop_size;

    // --- 3. 为并行计算预先计算偏移量 ---
    std::vector<int> split_offsets(split_dims.size() + 1, 0);
    for (size_t i = 0; i < split_dims.size(); ++i) {
        split_offsets[i + 1] = split_offsets[i] + split_dims[i];
    }

// --- 4. 并行处理 ---
#pragma omp parallel for
    for (size_t i = 0; i < out.size(); ++i) {
        void *out_ptr_void = out[i];
        const int split_size = split_dims[i];
        const int offset_in_dim = split_offsets[i];
        const DataType out_type = out_types[i];

        // 遍历所有不被分割的外部维度
        for (int outer_idx = 0; outer_idx < outer_loop_size; ++outer_idx) {
            // 计算源在当前外部维度块的基地址
            const float *src_base = origin + outer_idx * original_dim_size_at_split_axis + offset_in_dim * inner_loop_size;

            // --- 修改点：根据输出类型选择不同的处理路径 ---
            if (out_type == MLLM_TYPE_F32) {
                float *out_ptr = static_cast<float *>(out_ptr_void);
                float *dst_base = out_ptr + outer_idx * split_size * inner_loop_size;
                const size_t copy_bytes = split_size * inner_loop_size * sizeof(float);
                // memcpy(dst_base, src_base, copy_bytes);
                for (int split_idx = 0; split_idx < split_size; ++split_idx) {
                    const float *src = src_base + split_idx * inner_loop_size;
                    float *dst = dst_base + split_idx * inner_loop_size;
                    int count = inner_loop_size;
#if defined(__AVX__)
                    for (; count >= 8; count -= 8) {
                        __m256 data = _mm256_loadu_ps(src);
                        _mm256_storeu_ps(dst, data);
                        src += 8;
                        dst += 8;
                    }
#elif defined(__ARM_NEON)
                    for (; count >= 4; count -= 4) {
                        float32x4_t data = vld1q_f32(src);
                        vst1q_f32(dst, data);
                        src += 4;
                        dst += 4;
                    }
#endif
                    for (; count > 0; --count) *dst++ = *src++;
                }

            } else if (out_type == MLLM_TYPE_F16) {
                mllm_fp16_t *out_ptr = static_cast<mllm_fp16_t *>(out_ptr_void);
                mllm_fp16_t *dst_base = out_ptr + outer_idx * split_size * inner_loop_size;

                for (int split_idx = 0; split_idx < split_size; ++split_idx) {
                    const float *src = src_base + split_idx * inner_loop_size;
                    mllm_fp16_t *dst = dst_base + split_idx * inner_loop_size;
                    int count = inner_loop_size;

// 使用SIMD指令集进行高效的转换和内存拷贝
#if defined(__AVX__) && defined(__F16C__)
                    for (; count >= 8; count -= 8) {
                        // 从内存加载 8 个 float
                        __m256 float_vec = _mm256_loadu_ps(src);
                        // 将 8 个 float 转换为 8 个 fp16
                        __m128i fp16_vec = _mm256_cvtps_ph(float_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                        // 存储 8 个 fp16 (128 bits)
                        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), fp16_vec);
                        src += 8;
                        dst += 8;
                    }
#elif defined(__ARM_NEON)
                    for (; count >= 4; count -= 4) {
                        // 从内存加载 4 个 float
                        float32x4_t float_vec = vld1q_f32(src);
                        // 将 4 个 float 转换为 4 个 fp16
                        float16x4_t fp16_vec = vcvt_f16_f32(float_vec);
                        // 存储 4 个 fp16
                        vst1_f16(reinterpret_cast<float16_t *>(dst), fp16_vec);
                        src += 4;
                        dst += 4;
                    }
#endif

                    // 处理剩余不足一个SIMD寄存器大小的数据
                    for (; count > 0; --count) {
                        *dst++ = MLLM_FP32_TO_FP16(*src++);
                    }
                }
            } else {
                // 如果将来支持更多类型，可以在此添加
                // 为了安全，可以抛出异常或打印错误日志
            }
        }
    }
}