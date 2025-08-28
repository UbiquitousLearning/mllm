// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct LayerNormOpOptions : public BaseOpOptions<LayerNormOpOptions> {
  std::vector<int32_t> normalized_shape;
  bool elementwise_affine = true;
  bool bias = true;
  float eps = 1e-6;
};

class LayerNormOp : public BaseOp {
 public:
  explicit LayerNormOp(const LayerNormOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline const LayerNormOpOptions& options() const { return options_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  LayerNormOpOptions options_;
};

}  // namespace mllm::aops