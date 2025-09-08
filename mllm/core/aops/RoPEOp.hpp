// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct RoPEOpOptions : public BaseOpOptions<RoPEOpOptions> {
  float rope_theta = 10000.0F;
  int32_t max_position_embeddings = 16384;

  RoPEOpOptions() = default;
  explicit RoPEOpOptions(float theta, int32_t max_pos_embed) : rope_theta(theta), max_position_embeddings(max_pos_embed) {}
};

class RoPEOp : public BaseOp {
 public:
  explicit RoPEOp(const RoPEOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const RoPEOpOptions& options() const { return options_; }

 protected:
  RoPEOpOptions options_;
};

}  // namespace mllm::aops