// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/Conv3DOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

CPUConv3DOp::CPUConv3DOp(const aops::Conv3DOpOptions& options) : aops::Conv3DOp(options) {}

void CPUConv3DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto i_shape = input.shape();
  auto i_shape_size = i_shape.size();

  switch (options_.impl_type) {
    case aops::Conv3DOpImplType::kDefault: {
      switch (weight_.dtype()) {
        case kFloat32: {
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
          arm::conv3d_fp32_baseline(
              input.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(), bias_ ? bias_.ptr<mllm_fp32_t>() : nullptr,
              output.ptr<mllm_fp32_t>(), i_shape[i_shape_size - 5], i_shape[i_shape_size - 4], i_shape[i_shape_size - 3],
              i_shape[i_shape_size - 2], i_shape[i_shape_size - 1], options_.out_channels, options_.kernel_size[0],
              options_.kernel_size[1], options_.kernel_size[2], options_.stride[0], options_.stride[1], options_.stride[2]);
#endif
          break;
        }
        default: {
          NYI("Conv3D: unsupported weight data type");
          break;
        }
      }

      break;
    }
  }
}

}  // namespace mllm::cpu
