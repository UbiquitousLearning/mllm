// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/ReduceOps.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPUReduceMaxOp::CPUReduceMaxOp(const aops::ReduceMaxOpOptions& options) : aops::ReduceMaxOp(options) {}

void CPUReduceMaxOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();
  auto dim = options_.dim;
  auto keep_dim = options_.keep_dim;

  // Handle negative dimension index
  if (dim < 0) { dim += input.shape().size(); }

  // Special case: reduce over all dimensions
  if (dim == 0x7fffffff) {
    switch (dtype) {
      case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_max_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_max_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt8: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_max_int8(output.ptr<mllm_int8_t>(), input.ptr<mllm_int8_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_max_int16(output.ptr<mllm_int16_t>(), input.ptr<mllm_int16_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_max_int32(output.ptr<mllm_int32_t>(), input.ptr<mllm_int32_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      default: NYI("Unsupported data type for ReduceMaxOp");
    }
    return;
  }

  // For specific dimension reduction
  auto outer_size = 1;
  auto inner_size = 1;
  auto axis_size = input.shape()[dim];

  for (int i = 0; i < dim; ++i) { outer_size *= input.shape()[i]; }
  for (int i = dim + 1; i < input.shape().size(); ++i) { inner_size *= input.shape()[i]; }

  switch (dtype) {
    case kFloat32: {
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto output_ptr = output.ptr<mllm_fp32_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_max_fp32(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kFloat16: {
      auto input_ptr = input.ptr<mllm_fp16_t>();
      auto output_ptr = output.ptr<mllm_fp16_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_max_fp16(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt8: {
      auto input_ptr = input.ptr<mllm_int8_t>();
      auto output_ptr = output.ptr<mllm_int8_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_max_int8(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt16: {
      auto input_ptr = input.ptr<mllm_int16_t>();
      auto output_ptr = output.ptr<mllm_int16_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_max_int16(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt32: {
      auto input_ptr = input.ptr<mllm_int32_t>();
      auto output_ptr = output.ptr<mllm_int32_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_max_int32(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    default: NYI("Unsupported data type for ReduceMaxOp");
  }
}

CPUReduceMinOp::CPUReduceMinOp(const aops::ReduceMinOpOptions& options) : aops::ReduceMinOp(options) {}

void CPUReduceMinOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();
  auto dim = options_.dim;
  auto keep_dim = options_.keep_dim;

  // Handle negative dimension index
  if (dim < 0) { dim += input.shape().size(); }

  // Special case: reduce over all dimensions
  if (dim == 0x7fffffff) {
    switch (dtype) {
      case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_min_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_min_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt8: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_min_int8(output.ptr<mllm_int8_t>(), input.ptr<mllm_int8_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_min_int16(output.ptr<mllm_int16_t>(), input.ptr<mllm_int16_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_min_int32(output.ptr<mllm_int32_t>(), input.ptr<mllm_int32_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      default: NYI("Unsupported data type for ReduceMinOp");
    }
    return;
  }

  // For specific dimension reduction
  auto outer_size = 1;
  auto inner_size = 1;
  auto axis_size = input.shape()[dim];

  for (int i = 0; i < dim; ++i) { outer_size *= input.shape()[i]; }
  for (int i = dim + 1; i < input.shape().size(); ++i) { inner_size *= input.shape()[i]; }

  switch (dtype) {
    case kFloat32: {
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto output_ptr = output.ptr<mllm_fp32_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_min_fp32(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kFloat16: {
      auto input_ptr = input.ptr<mllm_fp16_t>();
      auto output_ptr = output.ptr<mllm_fp16_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_min_fp16(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt8: {
      auto input_ptr = input.ptr<mllm_int8_t>();
      auto output_ptr = output.ptr<mllm_int8_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_min_int8(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt16: {
      auto input_ptr = input.ptr<mllm_int16_t>();
      auto output_ptr = output.ptr<mllm_int16_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_min_int16(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt32: {
      auto input_ptr = input.ptr<mllm_int32_t>();
      auto output_ptr = output.ptr<mllm_int32_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_min_int32(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    default: NYI("Unsupported data type for ReduceMinOp");
  }
}

CPUReduceSumOp::CPUReduceSumOp(const aops::ReduceSumOpOptions& options) : aops::ReduceSumOp(options) {}

void CPUReduceSumOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();
  auto dim = options_.dim;
  auto keep_dim = options_.keep_dim;

  // Handle negative dimension index
  if (dim < 0) { dim += input.shape().size(); }

  // Special case: reduce over all dimensions
  if (dim == 0x7fffffff) {
    switch (dtype) {
      case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_sum_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_sum_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt8: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_sum_int8(output.ptr<mllm_int8_t>(), input.ptr<mllm_int8_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_sum_int16(output.ptr<mllm_int16_t>(), input.ptr<mllm_int16_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_sum_int32(output.ptr<mllm_int32_t>(), input.ptr<mllm_int32_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      default: NYI("Unsupported data type for ReduceSumOp");
    }
    return;
  }

  // For specific dimension reduction
  auto outer_size = 1;
  auto inner_size = 1;
  auto axis_size = input.shape()[dim];

  for (int i = 0; i < dim; ++i) { outer_size *= input.shape()[i]; }
  for (int i = dim + 1; i < input.shape().size(); ++i) { inner_size *= input.shape()[i]; }

  switch (dtype) {
    case kFloat32: {
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto output_ptr = output.ptr<mllm_fp32_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_sum_fp32(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kFloat16: {
      auto input_ptr = input.ptr<mllm_fp16_t>();
      auto output_ptr = output.ptr<mllm_fp16_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_sum_fp16(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt8: {
      auto input_ptr = input.ptr<mllm_int8_t>();
      auto output_ptr = output.ptr<mllm_int8_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_sum_int8(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                               axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt16: {
      auto input_ptr = input.ptr<mllm_int16_t>();
      auto output_ptr = output.ptr<mllm_int16_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_sum_int16(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt32: {
      auto input_ptr = input.ptr<mllm_int32_t>();
      auto output_ptr = output.ptr<mllm_int32_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_sum_int32(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    default: NYI("Unsupported data type for ReduceSumOp");
  }
}

CPUMeanOp::CPUMeanOp(const aops::MeanOpOptions& options) : aops::MeanOp(options) {}

void CPUMeanOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto dtype = input.dtype();
  auto dim = options_.dim;
  auto keep_dim = options_.keep_dim;

  // Handle negative dimension index
  if (dim < 0) { dim += input.shape().size(); }

  // Special case: reduce over all dimensions
  if (dim == 0x7fffffff) {
    switch (dtype) {
      case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_mean_fp32(output.ptr<mllm_fp32_t>(), input.ptr<mllm_fp32_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kFloat16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_mean_fp16(output.ptr<mllm_fp16_t>(), input.ptr<mllm_fp16_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt8: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_mean_int8(output.ptr<mllm_int8_t>(), input.ptr<mllm_int8_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt16: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_mean_int16(output.ptr<mllm_int16_t>(), input.ptr<mllm_int16_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      case kInt32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
        arm::reduce_mean_int32(output.ptr<mllm_int32_t>(), input.ptr<mllm_int32_t>(), 1, input.numel(), options_.getThreads());
#endif
        break;
      }
      default: NYI("Unsupported data type for MeanOp");
    }
    return;
  }

  // For specific dimension reduction
  auto outer_size = 1;
  auto inner_size = 1;
  auto axis_size = input.shape()[dim];

  for (int i = 0; i < dim; ++i) { outer_size *= input.shape()[i]; }
  for (int i = dim + 1; i < input.shape().size(); ++i) { inner_size *= input.shape()[i]; }

  switch (dtype) {
    case kFloat32: {
      auto input_ptr = input.ptr<mllm_fp32_t>();
      auto output_ptr = output.ptr<mllm_fp32_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_mean_fp32(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kFloat16: {
      auto input_ptr = input.ptr<mllm_fp16_t>();
      auto output_ptr = output.ptr<mllm_fp16_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_mean_fp16(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt8: {
      auto input_ptr = input.ptr<mllm_int8_t>();
      auto output_ptr = output.ptr<mllm_int8_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_mean_int8(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt16: {
      auto input_ptr = input.ptr<mllm_int16_t>();
      auto output_ptr = output.ptr<mllm_int16_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_mean_int16(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                 axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    case kInt32: {
      auto input_ptr = input.ptr<mllm_int32_t>();
      auto output_ptr = output.ptr<mllm_int32_t>();

      for (int out = 0; out < outer_size; ++out) {
        for (int in = 0; in < inner_size; ++in) {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::reduce_mean_int32(&output_ptr[out * inner_size + in], &input_ptr[out * axis_size * inner_size + in], inner_size,
                                 axis_size, options_.getThreads());
#endif
        }
      }
      break;
    }
    default: NYI("Unsupported data type for MeanOp");
  }
}

}  // namespace mllm::cpu
