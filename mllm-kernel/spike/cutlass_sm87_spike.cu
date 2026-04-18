/**
 * CUTLASS SM87 compilation spike.
 * Goal: verify CUTLASS int8 GEMM template can compile on SM87.
 *
 * Attempts to instantiate the same CUTLASS 2.x GEMM template that
 * sgl-kernel uses for SM80 (Ampere) int8 scaled matmul.
 */

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <iostream>

// Minimal instantiation matching sgl-kernel SM80 int8 GEMM config:
//   ElementA = int8_t, LayoutA = RowMajor
//   ElementB = int8_t, LayoutB = ColumnMajor
//   ElementC = float (accumulator), LayoutC = RowMajor
//   Epilogue: LinearCombination<float, 128/32, int32_t, float>
//   GemmShape<128, 128, 64>, WarpShape<64, 64, 64>, InstructionShape<16, 8, 32>

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = float;
using ElementAccumulator = int32_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,                          // output type
    128 / cutlass::sizeof_bits<ElementC>::value,  // elements per access
    ElementAccumulator,                // accumulator type
    float                              // compute type
>;

using GemmKernel = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,                 // A matrix
    ElementB, LayoutB,                 // B matrix
    ElementC, LayoutC,                 // C matrix
    ElementAccumulator,                // accumulator
    cutlass::arch::OpClassTensorOp,    // use Tensor Cores
    cutlass::arch::Sm80,               // target arch (SM80 codegen for SM87)
    cutlass::gemm::GemmShape<128, 128, 64>,     // thread block shape
    cutlass::gemm::GemmShape<64, 64, 64>,       // warp shape
    cutlass::gemm::GemmShape<16, 8, 32>,        // instruction shape (int8 tensor core)
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    5                                  // pipeline stages
>;

int main() {
    // Just verify template instantiation compiles.
    // Don't actually run - no GPU allocation needed for spike.
    GemmKernel gemm_op;

    std::cout << "CUTLASS SM87 spike: template instantiation SUCCESS" << std::endl;
    std::cout << "Kernel can_implement check would happen at runtime" << std::endl;
    return 0;
}
