// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct CastTypeOpOptions : public BaseOpOptions<CastTypeOpOptions> {
  DataTypes dtype;
};

class CastTypeOp : public BaseOp {
 public:
  explicit CastTypeOp(const CastTypeOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const CastTypeOpOptions& options() const { return options_; }

 protected:
  CastTypeOpOptions options_;
};

}  // namespace mllm::aops
