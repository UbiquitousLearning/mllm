// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/LayerNorm2DOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"

namespace mllm::cpu {

CPULayerNorm2DOp::CPULayerNorm2DOp(const aops::LayerNorm2DOpOptions& options) : aops::LayerNorm2DOp(options) {}

void CPULayerNorm2DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& i = inputs[0];
  auto& o = outputs[0];

  auto i_shape = i.shape();
  auto N = i_shape[0];
  auto C = i_shape[1];
  auto H = i_shape[2];
  auto W = i_shape[3];

  switch (i.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("Not impl for x86 64");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      arm::layernorm2d_fp32(i.ptr<mllm_fp32_t>(), weight_.ptr<mllm_fp32_t>(), bias_.ptr<mllm_fp32_t>(), o.ptr<mllm_fp32_t>(), N,
                            C, H, W, options_.eps);
#endif
      break;
    }
    default: {
      NYI("Not support data type");
    }
  }
}

}  // namespace mllm::cpu
