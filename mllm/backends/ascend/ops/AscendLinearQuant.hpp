// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "mllm/core/Tensor.hpp"

namespace mllm::ascend {

struct AscendLinearW8A8Artifacts {
  float scale_x = 0.0f;
  Tensor scale_w_cpu;
  Tensor scale_x_cpu;
  Tensor deq_scale_npu;
  Tensor deq_scale_w_npu;
  Tensor bias_int32_npu;
};

AscendLinearW8A8Artifacts prepareLinearW8A8Artifacts(const std::string& layer_name,
                                                     int out_channels,
                                                     const Tensor& scale_w_raw,
                                                     const Tensor& scale_x_raw);

}  // namespace mllm::ascend
