// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class InterpolateOpMode {
  kNearest,
  kLinear,
  kBilinear,
  kBicubic,
  kTrilinear,
};

struct InterpolateOpOptions : public BaseOpOptions<InterpolateOpOptions> {
  std::vector<int> size;
  std::vector<float> scale_factor;
  InterpolateOpMode mode = InterpolateOpMode::kNearest;
  bool align_corners = false;
  bool antialias = false;
};

class InterpolateOp : public BaseOp {
 public:
  explicit InterpolateOp(const InterpolateOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline InterpolateOpOptions& options() { return options_; }

 protected:
  InterpolateOpOptions options_;
};

}  // namespace mllm::aops
