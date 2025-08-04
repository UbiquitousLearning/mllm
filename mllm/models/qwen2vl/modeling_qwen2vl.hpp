// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/Nn.hpp"
#include "mllm/models/qwen2vl/configuration_qwen2vl.hpp"

namespace mllm::models {

class VisionMlp final : public nn::Module {
  int32_t dim_;
  int32_t hidden_dim_;

  nn::QuickGELU act_;
  nn::Linear fc_1_;
  nn::Linear fc_2_;

 public:
  VisionMlp() = default;

  inline explicit VisionMlp(const std::string& name, const Qwen2VLConfig& cfg) {
    dim_ = cfg.visual_embed_dim;
    hidden_dim_ = cfg.visual_embed_dim * cfg.visual_mlp_ratio;

    fc_1_ = reg<nn::Linear>("fc1", dim_, hidden_dim_);
    fc_2_ = reg<nn::Linear>("fc2", hidden_dim_, dim_);
    act_ = reg<nn::QuickGELU>("act");
  }

  std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override { return {fc_2_(act_(fc_1_(inputs[0])))}; }
};

}  // namespace mllm::models