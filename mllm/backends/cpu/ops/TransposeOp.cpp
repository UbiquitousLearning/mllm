// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/TransposeOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPUTransposeOp::CPUTransposeOp(const aops::TransposeOpOptions& options) : aops::TransposeOp(options) {}

void CPUTransposeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();

  int dim0 = options_.dim0;
  int dim1 = options_.dim1;

  if (dim0 < 0) dim0 += input.shape().size();
  if (dim1 < 0) dim1 += input.shape().size();

  auto input_shape = input.shape();

  // CASE 1. HW -> WH
  if (input_shape.size() == 2 && (dim0 + dim1 == 1)) {
    switch (dtype) {
      case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_hw_wh_fp32(input.ptr<mllm_fp32_t>(), output.ptr<mllm_fp32_t>(), input_shape[0], input_shape[1]);
#endif
        break;
      }
      case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_hw_wh_fp16(input.ptr<mllm_fp16_t>(), output.ptr<mllm_fp16_t>(), input_shape[0], input_shape[1]);
#endif
        break;
      }
      case kInt64: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_hw_wh_int64(input.ptr<mllm_int64_t>(), output.ptr<mllm_int64_t>(), input_shape[0], input_shape[1]);
#endif
        break;
      }
      default: NYI("Data type not supported");
    }
  }

  // CASE 2. BSHD -> BHSD
  else if (input.shape().size() == 4 && ((dim0 == 1 && dim1 == 2) || (dim0 == 2 && dim1 == 1))) {
    switch (dtype) {
      case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_bshd_bhsd_fp32(input.ptr<mllm_fp32_t>(), output.ptr<mllm_fp32_t>(), input_shape[0], input_shape[1],
                                      input_shape[2], input_shape[3]);
#endif
        break;
      }
      case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_bshd_bhsd_fp16(input.ptr<mllm_fp16_t>(), output.ptr<mllm_fp16_t>(), input_shape[0], input_shape[1],
                                      input_shape[2], input_shape[3]);
#endif
        break;
      }
      case kInt64: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_bshd_bhsd_int64(input.ptr<mllm_int64_t>(), output.ptr<mllm_int64_t>(), input_shape[0], input_shape[1],
                                       input_shape[2], input_shape[3]);
#endif
        break;
      }
      default: NYI("Data type not supported");
    }
  }

  // CASE 3. Exchange last 2 dims.
  // FIXME: (dim0 + dim1 == 1) is error logic.
  else if (input_shape.size() != 2 && (dim0 + dim1 == 1)) {
    int batch = 0;
    for (int i = 0; i < input_shape.size() - 2; i++) { batch *= input_shape[i]; }

    switch (dtype) {
      case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_last_dims_fp32(input.ptr<mllm_fp32_t>(), output.ptr<mllm_fp32_t>(), batch, input_shape[0],
                                      input_shape[1]);
#endif
        break;
      }
      case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_last_dims_fp16(input.ptr<mllm_fp16_t>(), output.ptr<mllm_fp16_t>(), batch, input_shape[0],
                                      input_shape[1]);
#endif
        break;
      }
      case kInt64: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::transpose_last_dims_int64(input.ptr<mllm_int64_t>(), output.ptr<mllm_int64_t>(), batch, input_shape[0],
                                       input_shape[1]);
#endif
        break;
      }
      default: NYI("Data type not supported");
    }
  }

  // CASE 4. General permute
  else {
    std::vector<int32_t> permute_axis(input_shape.size());
    for (int i = 0; i < input_shape.size(); i++) { permute_axis[i] = i; }

    std::swap(permute_axis[dim0], permute_axis[dim1]);

    switch (dtype) {
      case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::permute_fp32(input.ptr<mllm_fp32_t>(), output.ptr<mllm_fp32_t>(), input_shape.data(), permute_axis.data(),
                          permute_axis.size());
#endif
        break;
      }
      case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::permute_fp16(input.ptr<mllm_fp16_t>(), output.ptr<mllm_fp16_t>(), input_shape.data(), permute_axis.data(),
                          permute_axis.size());
#endif
        break;
      }
      case kInt64: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::permute_generic<mllm_int64_t>(input.ptr<mllm_int64_t>(), output.ptr<mllm_int64_t>(), input_shape.data(),
                                           permute_axis.data(), permute_axis.size());
#endif
        break;
      }
      default: NYI("Data type not supported");
    }
  }
}

}  // namespace mllm::cpu
