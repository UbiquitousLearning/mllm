// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/ViewOp.hpp"

namespace mllm::opencl {

class OpenCLViewOp final : public aops::ViewOp {
 public:
  explicit OpenCLViewOp(const aops::ViewOpOptions& options);
};

class OpenCLViewOpFactory : public TypedOpFactory<OpTypes::kView, aops::ViewOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ViewOpOptions& options) override {
    return std::make_shared<OpenCLViewOp>(options);
  }
};

}  // namespace mllm::opencl
