// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/Tensor.hpp"

namespace mllm::models::qwen2vl {

struct Qwen2VLForCausalLMOutputPast {
  Tensor sequence = Tensor::nil();
  Tensor img = Tensor::nil();
  Tensor grid_thw = Tensor::nil();
  Tensor position_ids = Tensor::nil();
  bool has_visual = false;
};

}  // namespace mllm::models::qwen2vl
