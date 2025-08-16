// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SliceOp.hpp"

namespace mllm::cpu {

class CPUSliceOp final : public aops::SliceOp {
 public:
  explicit CPUSliceOp(const aops::SliceOpOptions& options);
};

class CPUSliceOpFactory : public TypedOpFactory<OpTypes::kSlice, aops::SliceOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SliceOpOptions& options) override {
    return std::make_shared<CPUSliceOp>(options);
  }
};

}  // namespace mllm::cpu
