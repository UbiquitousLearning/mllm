// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <atb/types.h>

namespace mllm::ascend {

atb::Status executeAscendCausalMaskKernel(const atb::Tensor& input,
                                          atb::Tensor& output,
                                          bool sliding_window,
                                          int32_t window_size);

}  // namespace mllm::ascend
