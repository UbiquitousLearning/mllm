// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "mllm/core/Tensor.hpp"

namespace mllm::ascend {

void runLinearDynamicW8A8Eager(const std::string& layer_name,
                               const Tensor& x,
                               const Tensor& weight,
                               const Tensor& bias_int32_npu,
                               const Tensor& deq_scale_w_npu,
                               Tensor& y);

}  // namespace mllm::ascend
