/**
 * CUTLASS SM87 spike #2: SM89 tile shapes (100K shared memory config).
 * SM87 (Jetson Orin) has ~100K shared memory, same as SM86/89.
 * sgl-kernel uses smaller tiles for SM89 to fit in 100K smem.
 *
 * Tests multiple tile configurations from sgl-kernel sm89_dispatch_shape.
 */

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <iostream>

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = float;
using ElementAccumulator = int32_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

template<int TM, int TN, int TK, int WM, int WN, int WK, int Stages>
using GemmType = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TM, TN, TK>,
    cutlass::gemm::GemmShape<WM, WN, WK>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stages
>;

int main() {
    // SM89 tile configs from sgl-kernel (smaller smem footprint)
    // Config 1: M<=16
    GemmType<16, 64, 128, 16, 64, 64, 4> gemm1;
    std::cout << "SM89 tile (16,64,128) stages=4: OK" << std::endl;

    // Config 2: M<=32
    GemmType<32, 64, 128, 32, 64, 64, 4> gemm2;
    std::cout << "SM89 tile (32,64,128) stages=4: OK" << std::endl;

    // Config 3: M<=64
    GemmType<64, 64, 128, 64, 64, 64, 3> gemm3;
    std::cout << "SM89 tile (64,64,128) stages=3: OK" << std::endl;

    // Config 4: M>64 (large tiles)
    GemmType<128, 64, 64, 64, 64, 64, 3> gemm4;
    std::cout << "SM89 tile (128,64,64) stages=3: OK" << std::endl;

    // SM80 large tile for comparison (might exceed SM87 smem)
    GemmType<128, 128, 64, 64, 64, 64, 5> gemm5_sm80;
    std::cout << "SM80 tile (128,128,64) stages=5: compiled (smem may exceed at runtime)" << std::endl;

    std::cout << "\nAll tile configurations compiled successfully for SM87!" << std::endl;
    return 0;
}
