/**
 * CUTLASS INT8 Scaled MatMul for SM80+ (Ampere).
 *
 * Ported from sglang sgl-kernel/csrc/gemm/int8_gemm_kernel.cu
 * Adapted for mllm-kernel with SM87 (Jetson Orin) support.
 *
 * Only includes CUTLASS 2.x paths (SM80/87/89). No SM90 (Hopper) support.
 */

#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/epilogue_with_visitor.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/numeric_types.h>
#include <torch/extension.h>

#include "cutlass_extensions/epilogue/epilogue_per_row_per_col_scale.h"
#include "cutlass_extensions/gemm/gemm_universal_base_compat.h"
#include "cutlass_extensions/gemm/gemm_with_epilogue_visitor.h"

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

inline int getSMVersion() {
  int device{-1};
  cudaGetDevice(&device);
  int sm_major = 0, sm_minor = 0;
  cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
  return sm_major * 10 + sm_minor;
}

// ---------------------------------------------------------------------------
// Core CUTLASS GEMM template (CUTLASS 2.x with per-row/col scale epilogue)
// ---------------------------------------------------------------------------

template <
    typename ElementOutput,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    int NumStages>
void cutlass_int8_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementInputA = int8_t;
  using ElementInputB = int8_t;

  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  using DefaultGemmConf = cutlass::gemm::device::DefaultGemmConfiguration<
      OperatorClass, ArchTag, ElementInputA, ElementInputB,
      ElementOutput, ElementCompute>;
  using EpilogueOutputOp = typename DefaultGemmConf::EpilogueOutputOp;

  using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
      ElementInputA, cutlass::layout::RowMajor, DefaultGemmConf::kAlignmentA,
      ElementInputB, cutlass::layout::ColumnMajor, DefaultGemmConf::kAlignmentB,
      ElementOutput, cutlass::layout::RowMajor,
      ElementAccumulator,
      OperatorClass, ArchTag,
      ThreadblockShape, WarpShape, InstructionShape,
      EpilogueOutputOp, ThreadblockSwizzle, NumStages,
      true, typename DefaultGemmConf::Operator>::GemmKernel;

  using AlphaColTileIterator =
      cutlass::epilogue::threadblock::PredicatedTileIterator<
          cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
              typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Shape,
              typename GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::Count,
              GemmKernel_::Epilogue::OutputTileIterator::ThreadMap::kThreads,
              GemmKernel_::Epilogue::OutputTileIterator::kElementsPerAccess,
              cutlass::sizeof_bits<ElementOutput>::value>,
          ElementCompute>;

  using EpilogueVisitor =
      typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerCol<
          ThreadblockShape,
          GemmKernel_::kThreadCount,
          AlphaColTileIterator,
          typename GemmKernel_::Epilogue::OutputTileIterator,
          ElementAccumulator, ElementCompute, EpilogueOutputOp>;

  using Epilogue = typename cutlass::epilogue::threadblock::
      EpilogueWithVisitorFromExistingEpilogue<
          EpilogueVisitor, typename GemmKernel_::Epilogue>::Epilogue;

  using GemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
      typename GemmKernel_::Mma, Epilogue, ThreadblockSwizzle>;

  using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

  Gemm gemm_op;

  int m = mat_a.size(0);
  int k = mat_a.size(1);
  int n = mat_b.size(1);

  auto a_ptr = static_cast<ElementInputA*>(mat_a.data_ptr());
  auto b_ptr = static_cast<ElementInputB*>(mat_b.data_ptr());
  auto o_ptr = static_cast<ElementOutput*>(out.data_ptr());
  auto a_s_ptr = static_cast<ElementCompute*>(scales_a.data_ptr());
  auto b_s_ptr = static_cast<ElementCompute*>(scales_b.data_ptr());

  int64_t lda = mat_a.stride(0);
  int64_t ldb = mat_b.stride(1);
  int64_t ldd = out.stride(0);

  ElementOutput* bias_ptr = nullptr;
  int64_t ldc = 0;
  if (bias) {
    bias_ptr = static_cast<ElementOutput*>(bias->data_ptr());
  }

  typename EpilogueOutputOp::Params linearScalingParams;
  typename EpilogueVisitor::Arguments visitor_args{linearScalingParams};

  typename Gemm::Arguments args{
      {m, n, k},
      {a_ptr, lda}, {b_ptr, ldb},
      {b_s_ptr, 0}, {a_s_ptr, 0},
      {bias_ptr, ldc}, {o_ptr, ldd},
      visitor_args};

  auto workspace = torch::empty(
      gemm_op.get_workspace_size(args),
      torch::TensorOptions().dtype(torch::kUInt8).device(mat_a.device()));

  auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(
      can_implement == cutlass::Status::kSuccess,
      "CUTLASS can_implement failed: ",
      cutlassGetStatusString(can_implement));

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS execution failed: ",
      cutlassGetStatusString(status));
}

// ---------------------------------------------------------------------------
// Dispatch shape for sm89 (L40S, L20, RTX 4090), according to:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/scaled_mm_c2x_sm89_int8_dispatch.cuh
// ---------------------------------------------------------------------------

template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm89_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 16) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<16, 128, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape, 4>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 32) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<32, 128, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape, 4>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape, 3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    if (n <= 8192) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape, 3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 256) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape, 3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 8192) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<256, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape, 3>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else {
    cutlass_int8_scaled_mm<ElementOutput, ArchTag,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

// ---------------------------------------------------------------------------
// SM80 dispatch (160K shared memory, for SM80/SM87)
// ---------------------------------------------------------------------------

template <typename ElementOutput, typename ArchTag, typename InstructionShape>
void sm80_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  int m = mat_a.size(0);
  int n = mat_b.size(1);
  if (m <= 16) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape, 6>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 32) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape, 6>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    if (n <= 4096) {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      cutlass_int8_scaled_mm<ElementOutput, ArchTag,
          cutlass::gemm::GemmShape<64, 128, 128>,
          cutlass::gemm::GemmShape<64, 64, 64>,
          InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
  } else if (m <= 128 && n < 8192) {
    cutlass_int8_scaled_mm<ElementOutput, ArchTag,
        cutlass::gemm::GemmShape<64, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  } else {
    cutlass_int8_scaled_mm<ElementOutput, ArchTag,
        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        InstructionShape, 5>(out, mat_a, mat_b, scales_a, scales_b, bias);
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

torch::Tensor int8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const std::string& out_dtype_str,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be 2D");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be 2D");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be row-major");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_b must be column-major");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "shape mismatch");
  TORCH_CHECK(mat_a.size(1) % 16 == 0, "K must be multiple of 16");
  TORCH_CHECK(mat_b.size(1) % 8 == 0, "N must be multiple of 8");
  TORCH_CHECK(mat_a.scalar_type() == torch::kInt8, "mat_a must be Int8");
  TORCH_CHECK(mat_b.scalar_type() == torch::kInt8, "mat_b must be Int8");
  TORCH_CHECK(scales_a.numel() == mat_a.size(0), "scales_a size mismatch");
  TORCH_CHECK(scales_b.numel() == mat_b.size(1), "scales_b size mismatch");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be fp32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be fp32");

  torch::Dtype out_dtype;
  if (out_dtype_str == "float16") {
    out_dtype = torch::kHalf;
  } else if (out_dtype_str == "bfloat16") {
    out_dtype = torch::kBFloat16;
  } else {
    TORCH_CHECK(false, "out_dtype must be 'float16' or 'bfloat16', got: ", out_dtype_str);
  }

  if (bias) {
    TORCH_CHECK(bias->numel() == mat_b.size(1), "bias size mismatch");
    TORCH_CHECK(bias->dtype() == out_dtype, "bias dtype must match out_dtype");
  }

  auto out = torch::empty(
      {mat_a.size(0), mat_b.size(1)},
      mat_a.options().dtype(out_dtype));

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  using ArchTag = cutlass::arch::Sm80;

  // SM86/SM89 have smaller shared memory and use sglang's SM89 tile shapes.
  // SM87 (Jetson Orin) has 164K smem, same as SM80, so it stays on SM80.
  int sm_version = getSMVersion();

  if (sm_version >= 80 && sm_version < 90) {
    if (sm_version == 86 || sm_version == 89) {
      if (out_dtype == torch::kBFloat16) {
        sm89_dispatch_shape<cutlass::bfloat16_t, ArchTag, InstructionShape>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      } else {
        sm89_dispatch_shape<cutlass::half_t, ArchTag, InstructionShape>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      }
    } else {
      if (out_dtype == torch::kBFloat16) {
        sm80_dispatch_shape<cutlass::bfloat16_t, ArchTag, InstructionShape>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      } else {
        sm80_dispatch_shape<cutlass::half_t, ArchTag, InstructionShape>(
            out, mat_a, mat_b, scales_a, scales_b, bias);
      }
    }
  } else {
    TORCH_CHECK(false, "Unsupported SM version: ", sm_version, ". Requires SM80-SM89.");
  }

  return out;
}

// ---------------------------------------------------------------------------
// PyBind11 binding
// ---------------------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int8_scaled_mm", &int8_scaled_mm,
        "CUTLASS INT8 scaled matmul with per-row/col scaling",
        py::arg("mat_a"), py::arg("mat_b"),
        py::arg("scales_a"), py::arg("scales_b"),
        py::arg("out_dtype"), py::arg("bias") = py::none());
}
