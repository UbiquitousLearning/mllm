// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class Conv1DOpImplType {
  kDefault = 0,
};

struct Conv1DOpOptions : public BaseOpOptions<Conv1DOpOptions> {
  int32_t in_channels;
  int32_t out_channels;
  int32_t kernel_size;
  int32_t stride;
  bool bias = true;
  int32_t padding = 0;
  int32_t groups = 1;
  int32_t dilation = 1;
};

class Conv1DOp : public BaseOp {
 public:
  explicit Conv1DOp(const Conv1DOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline Conv1DOpOptions& options() { return options_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  Conv1DOpOptions options_;
};

}  // namespace mllm::aops
