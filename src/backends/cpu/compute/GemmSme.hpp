#ifndef SMEGEMM_HPP
#define SMEGEMM_HPP

#include <iostream>
#include <vector>
#include <cstdint>
#include <memory> // For std::unique_ptr

// UNUSED

// 定义 mllm_fp16_t 和相关宏/函数
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__ARM_NEON_FP16_TYPE) || defined(__clang__)
#define MLLM_FP16_SUPPORTED 1
#else
#define MLLM_FP16_SUPPORTED 0
#endif

#if defined(__ARM_FEATURE_SME)
#include <arm_sme.h> // 包含SME intrinsics所必需的头文件

// SME Kernel for F32 * F32 -> F32 (output)
// A is M x K (row-major), B is K x N (row-major)
// Computes C(M,N) = A * B
// lda, ldb, ldc are element strides (number of elements to get to the next row)
__attribute__((target("sme"))) void sme_gemm_f32f32_f32(
    const float *A, const float *B, float *C,
    int M, int K, int N,
    int lda, int ldb, int ldc) {
    uint64_t vl_f32 = svcntw();

    if (M == 0 || K == 0 || N == 0 || vl_f32 == 0) {
        return;
    }

    const int vm_size = vl_f32;
    const int vn_size = vl_f32;

    svbool_t pg_all_f32 = svptrue_b32();

    for (int m_base = 0; m_base < M; m_base += vm_size) {
        uint64_t current_m_tile_height = (M - m_base < vm_size) ? (M - m_base) : vm_size;
        svbool_t pg_m = svwhilelt_b32_u32(0, (uint32_t)current_m_tile_height);

        for (int n_base = 0; n_base < N; n_base += vn_size) {
            uint64_t current_n_tile_width = (N - n_base < vn_size) ? (N - n_base) : vn_size;
            svbool_t pg_n = svwhilelt_b32_u32(0, (uint32_t)current_n_tile_width);

            __arm_sme_zero_za_tiles(1 << 0);

            for (int k_iter = 0; k_iter < K; ++k_iter) {
                svfloat32_t sve_a_col_k;
                if (current_m_tile_height == vm_size && lda == K /* implies elements are contiguous if lda is K and we process full column for tile */) {
                    // This gather is for non-contiguous elements if lda != 1 (for a conceptual column view)
                    // If lda == K, A[m_base*lda + k_iter] is A[m_base*K + k_iter].
                    // svld1_gather_offset_f32 expects base + vector_offsets.
                    // For A[m_base : m_base+vm_size-1, k_iter]
                    // Base is A + k_iter (start of k-th column if stored column major)
                    // OR A + m_base*lda + k_iter (element A[m_base, k_iter])
                    // Offsets are 0, lda, 2*lda ... for elements in the same column
                    sve_a_col_k = svld1_gather_offset_f32(pg_m, A + m_base * lda + k_iter, svmul_n_u32_z(pg_m, svindex_u32(0, 1), (uint32_t)lda));

                } else {
                    float temp_a_col[vm_size];
                    for (uint64_t i = 0; i < current_m_tile_height; ++i) temp_a_col[i] = A[(m_base + i) * lda + k_iter];
                    for (uint64_t i = current_m_tile_height; i < vm_size; ++i) temp_a_col[i] = 0.0f;
                    sve_a_col_k = svld1_f32(pg_all_f32, temp_a_col);
                }

                svfloat32_t sve_b_row_k = svld1_f32(pg_n, B + k_iter * ldb + n_base);
                svfmopa_za_f32_m(pg_all_f32, 0, sve_a_col_k, sve_b_row_k);
            }

            for (uint64_t m_local_idx = 0; m_local_idx < current_m_tile_height; ++m_local_idx) {
                svfloat32_t acc_f32_vec = __arm_sme_read_za32_vert_slice(0, m_local_idx, pg_all_f32);
                svst1_f32(pg_n, C + (m_base + m_local_idx) * ldc + n_base, acc_f32_vec);
            }
        }
    }
}

// SME Kernel for F16 * F16 -> F32 (output)
// A is M x K (row-major, mllm_fp16_t), B is K x N (row-major, mllm_fp16_t)
// Computes C(M,N) = A * B, with C being float
// lda, ldb, ldc are element strides
__attribute__((target("sme"))) void sme_gemm_f16f16_f32(
    const mllm_fp16_t *A, const mllm_fp16_t *B, float *C,
    int M, int K, int N,
    int lda, int ldb, int ldc) {
    uint64_t vl_f16 = svcnth(); // Number of __fp16 elements in an SVE vector (SVE Halfword count)

    if (M == 0 || K == 0 || N == 0 || vl_f16 == 0) {
        // std::cerr << "SME GEMM F16: Invalid dimensions or SVE length 0." << std::endl;
        return;
    }

    // --- Tiling Parameters (Highly Simplified) ---
    const int vm_size = vl_f16; // Tile height for C and A (in terms of FP16 elements)
    const int vn_size = vl_f16; // Tile width for C and B (in terms of FP16 elements)

    svbool_t pg_all_f16 = svptrue_b16(); // Predicate all true for 16-bit elements

    // Loop over M in tiles of vm_size
    for (int m_base = 0; m_base < M; m_base += vm_size) {
        uint64_t current_m_tile_height = (M - m_base < vm_size) ? (M - m_base) : vm_size;
        svbool_t pg_m = svwhilelt_b16_u16(0, (uint16_t)current_m_tile_height);

        // Loop over N in tiles of vn_size
        for (int n_base = 0; n_base < N; n_base += vn_size) {
            uint64_t current_n_tile_width = (N - n_base < vn_size) ? (N - n_base) : vn_size;
            svbool_t pg_n = svwhilelt_b16_u16(0, (uint16_t)current_n_tile_width);

            // 1. Zero the ZA tile (or relevant slices) that will hold the C_tile.
            //    ZA accumulates in FP32 when using svfmopa_za32_f16_m.
            __arm_sme_zero_za_tiles(1 << 0); // Zero ZA tile 0

            // 2. Loop over the K dimension
            for (int k_iter = 0; k_iter < K; ++k_iter) {
                // Load a column vector from A's current tile: A[m_base : m_base+vm_size-1, k_iter]
                svfloat16_t sve_a_col_k;
                // Simplified loading (as in F32 version, adapt for mllm_fp16_t)
                // Production code would use efficient gather/strided loads for __fp16.
                if (current_m_tile_height == vm_size) {
                    sve_a_col_k = svld1_gather_offset_f16(pg_m, A + k_iter, svmul_n_u32_z(svptrue_b32() /* SVE index operates on u32 usually */, svindex_u32(0, 1), lda));

                } else {
                    mllm_fp16_t temp_a_col[vm_size];
                    for (uint64_t i = 0; i < current_m_tile_height; ++i) temp_a_col[i] = A[(m_base + i) * lda + k_iter];
                    for (uint64_t i = current_m_tile_height; i < vm_size; ++i) temp_a_col[i] = MLLM_FP32_TO_FP16(0.0f); // Padding
                    sve_a_col_k = svld1_f16(pg_all_f16, temp_a_col);
                }

                // Load a row vector from B's current tile: B[k_iter, n_base : n_base+vn_size-1]
                svfloat16_t sve_b_row_k = svld1_f16(pg_n, B + k_iter * ldb + n_base);

                // Perform FP16 outer product and accumulate into FP32 ZA tile 0.
                // void svfmopa_za32_f16_m(svbool_t pg_op, uint8_t tile_idx, svfloat16_t zn_col, svfloat16_t zm_row);
                svfmopa_za32_f16_m(pg_all_f16, /*ZA tile index*/ 0, sve_a_col_k, sve_b_row_k);
            }

            // 3. Read the accumulated FP32 result from ZA tile 0 and store it to C_tile (which is float *C)
            //    A ZA tile (when accumulated from FP16 to FP32) has svcntw() "rows",
            //    each row being an SVE FP32 vector.
            //    The number of FP16 elements per SVE vector is vl_f16 (svcnth()).
            //    The number of FP32 elements per SVE vector is vl_f32 (svcntw()).
            //    When reading ZA after svfmopa_za32_f16_m, we read FP32 vectors.
            //    The ZA tile is conceptually vl_f16 x vl_f16 of FP16 elements, but stored/read as FP32.
            //    This means __arm_sme_read_za32_vert_slice reads vl_f32 elements.
            //    The m_local_idx should iterate up to where FP16 elements were conceptually written.

            svbool_t pg_all_f32 = svptrue_b32(); // Predicate for reading/writing FP32
            for (uint64_t m_local_idx = 0; m_local_idx < current_m_tile_height; ++m_local_idx) {
                // The ZA tile stores results as FP32.
                // We need to read 'current_m_tile_height' conceptual rows.
                // Each conceptual row from MOPA perspective was 'vl_f16' FP16 elements tall.
                // When reading as FP32, we read 'vl_f32' elements per SVE vector.
                // This mapping needs to be precise.
                // If svfmopa_za32_f16_m populates a vl_f16 x vl_f16 logical FP16 tile within ZA (stored as FP32),
                // then reading it slice by slice...

                // __arm_sme_read_za32_vert_slice reads an SVE vector (vl_f32 elements)
                // from a "vertical SVE slice" (row) of the ZA tile.
                // The number of such rows in ZA is also vl_f32 (for a square ZA tile view).
                // We need to store `current_m_tile_height` rows of C, each with `current_n_tile_width` FP32 elements.
                // The `m_local_idx` corresponds to the row index in the current C tile.

                if (m_local_idx < svcntw()) { // Ensure we don't read more ZA rows than available
                                              // This condition might need refinement based on how MOPA maps to ZA
                    svfloat32_t acc_f32_vec = __arm_sme_read_za32_vert_slice(0, m_local_idx, pg_all_f32);

                    // Store the FP32 SVE vector to the C matrix, predicated by current_n_tile_width
                    // The predicate pg_n was for FP16. We need an FP32 predicate for storing FP32 vector.
                    svbool_t pg_n_f32 = svwhilelt_b32_u32(0, (uint32_t)current_n_tile_width);
                    svst1_f32(pg_n_f32, C + (m_base + m_local_idx) * ldc + n_base, acc_f32_vec);
                } else if (current_m_tile_height > svcntw() && MLLM_FP16_SUPPORTED) {
                    // This case is more complex: current output tile height (in FP16 elements)
                    // is larger than one SVE vector of FP32 elements can represent as rows.
                    // This means the ZA tile was conceptually wider or taller in FP16 terms than
                    // what svcntw() rows of svcntw() FP32 elements implies.
                    // For now, this simplified example might not correctly handle tiles significantly larger
                    // than svcntw() x svcntw() FP32 elements directly without more complex ZA indexing.
                    // The mapping of a vl_f16 x vl_f16 conceptual FP16 tile into an FP32 ZA tile
                    // means the ZA tile stores (vl_f16 * vl_f16) FP16 products, summed as FP32.
                    // The __arm_sme_read_za32_vert_slice reads vl_f32 elements.
                    // We need to ensure m_local_idx correctly maps to the ZA slice.
                    // If vl_f16 = 2 * vl_f32 (e.g. 256-bit SVE), one FP16 row matches half an FP32 row.
                    // This part requires careful thought on ZA layout for FP16 MOPA into FP32 ZA.
                    // For simplicity, this example assumes current_m_tile_height <= svcntw().
                }
            }
        }
    }
}
#endif // __ARM_FEATURE_SME

#endif // SMEGEMM_HPP