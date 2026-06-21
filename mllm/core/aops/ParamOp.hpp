// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct ParamOpOptions : public BaseOpOptions<ParamOpOptions> {
  std::string name;
  Tensor::shape_t shape;
};

class ParamOp : public BaseOp {
 public:
  explicit ParamOp(const ParamOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }

  inline const ParamOpOptions& options() const { return options_; }

 protected:
  Tensor weight_;
  ParamOpOptions options_;
};

}  // namespace mllm::aops