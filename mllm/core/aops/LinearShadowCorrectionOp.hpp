// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct LinearShadowCorrectionOpOptions : public BaseOpOptions<LinearShadowCorrectionOpOptions> {
  int32_t in_channels = 0;
  int32_t out_channels = 0;
  int32_t activation_bits = 16;
};

// Restores the unclipped tails of a per-tensor symmetric quantized linear.
//
// Inputs:
//   0. Original Float32 linear input (last dimension in_channels), or a complete
//      Int32 correction packet for each 256-row tile of the 8960 x 1536 projection.
//   1. Saturated signed quantized linear output; its last dimension is
//      out_channels.
//   2. Float32 baseline after dequantization and residual addition.
// Output:
//   A fresh Float32 tensor with the same shape as input 2.
//
// A layer named "<linear>.shadow" consumes the existing parameters owned by
// "<linear>". It does not add or rename checkpoint tensors.
class LinearShadowCorrectionOp : public BaseOp {
 public:
  explicit LinearShadowCorrectionOp(const LinearShadowCorrectionOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  [[nodiscard]] const LinearShadowCorrectionOpOptions& options() const { return options_; }

 protected:
  [[nodiscard]] std::string parameterPrefix() const;

  LinearShadowCorrectionOpOptions options_;
  Tensor weight_;
  Tensor weight_scale_;
  Tensor input_scale_;
  Tensor output_scale_;
  Tensor clip_input_;
  Tensor clip_output_;
};

}  // namespace mllm::aops
