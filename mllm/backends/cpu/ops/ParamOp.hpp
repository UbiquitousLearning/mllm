// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ParamOp.hpp"

namespace mllm::cpu {

class CPUParamOp final : public aops::ParamOp {
 public:
  explicit CPUParamOp(const aops::ParamOpOptions& options);
};

class CPUParamOpFactory : public TypedOpFactory<OpTypes::kParam, aops::ParamOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ParamOpOptions& options) override {
    return std::make_shared<CPUParamOp>(options);
  }
};

}  // namespace mllm::cpu
