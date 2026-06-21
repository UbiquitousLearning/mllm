// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct X2XOpOptions : public BaseOpOptions<X2XOpOptions> {
  DeviceTypes device;
};

class X2XOp : public BaseOp {
 public:
  explicit X2XOp(const X2XOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const X2XOpOptions& options() const { return options_; }

 protected:
  X2XOpOptions options_;
};

}  // namespace mllm::aops