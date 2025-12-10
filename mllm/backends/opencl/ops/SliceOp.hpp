// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/SliceOp.hpp"

namespace mllm::opencl {

class OpenCLSliceOp final : public aops::SliceOp {
 public:
  explicit OpenCLSliceOp(const aops::SliceOpOptions& options);
};

class OpenCLSliceOpFactory : public TypedOpFactory<OpTypes::kSlice, aops::SliceOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SliceOpOptions& options) override {
    return std::make_shared<OpenCLSliceOp>(options);
  }
};

}  // namespace mllm::opencl
