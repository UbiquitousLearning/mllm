// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct AvgPool1dOpOptions : public BaseOpOptions<AvgPool1dOpOptions> {
  int32_t kernel_size;
  int32_t stride;
  int32_t padding = 0;
  bool ceil_mode = false;
  bool count_include_pad = true;
};

class AvgPool1dOp : public BaseOp {
 public:
  explicit AvgPool1dOp(const AvgPool1dOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline AvgPool1dOpOptions& options() { return options_; }

 protected:
  AvgPool1dOpOptions options_;
};

}  // namespace mllm::aops
