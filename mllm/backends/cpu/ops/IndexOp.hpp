// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/IndexOp.hpp"

namespace mllm::cpu {

class CPUIndexOp final : public aops::IndexOp {
 public:
  explicit CPUIndexOp(const aops::IndexOpOptions& options);
};

class CPUIndexOpFactory : public TypedOpFactory<OpTypes::kIndex, aops::IndexOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::IndexOpOptions& options) override {
    return std::make_shared<CPUIndexOp>(options);
  }
};

}  // namespace mllm::cpu
