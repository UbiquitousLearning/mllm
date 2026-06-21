// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct LayerNorm2DOpOptions : public BaseOpOptions<LayerNorm2DOpOptions> {
  int32_t num_channels;
  float eps = 1e-6;
};

class LayerNorm2DOp : public BaseOp {
 public:
  explicit LayerNorm2DOp(const LayerNorm2DOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline LayerNorm2DOpOptions& options() { return options_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  LayerNorm2DOpOptions options_;
};

}  // namespace mllm::aops
