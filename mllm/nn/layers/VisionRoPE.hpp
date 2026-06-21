#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/VisionRoPEOp.hpp"

namespace mllm::nn {

class VisionRoPE : public Layer {
 public:
  VisionRoPE();

  explicit VisionRoPE(const aops::VisionRoPEOpOptions& Options);

  VisionRoPE(const aops::VisionRoPEOpOptionsType type, const aops::Qwen2VLRoPEOpOptions& Options);

  MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
