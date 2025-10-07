// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class RoPEOpOptionsInputType : uint8_t {
  kBHSD = 0,
  kBSHD = 1,
};

struct RoPEOpOptions : public BaseOpOptions<RoPEOpOptions> {
  float rope_theta = 10000.0F;
  int32_t max_position_embeddings = 16384;
  RoPEOpOptionsInputType input_type = RoPEOpOptionsInputType::kBHSD;
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
