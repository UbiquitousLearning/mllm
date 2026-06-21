// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/GraphOps.hpp"

namespace mllm::opencl {

class OpenCLGraphBeginOp final : public aops::GraphBeginOp {
 public:
  explicit OpenCLGraphBeginOp(const aops::GraphBeginOpOptions& options);
};

class OpenCLGraphBeginOpFactory final : public TypedOpFactory<OpTypes::kGraphBegin, aops::GraphBeginOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GraphBeginOpOptions& options) override {
    return std::make_shared<OpenCLGraphBeginOp>(options);
  }
};

class OpenCLGraphEndOp final : public aops::GraphEndOp {
 public:
  explicit OpenCLGraphEndOp(const aops::GraphEndOpOptions& options);
};

class OpenCLGraphEndOpFactory final : public TypedOpFactory<OpTypes::kGraphEnd, aops::GraphEndOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GraphEndOpOptions& options) override {
    return std::make_shared<OpenCLGraphEndOp>(options);
  }
};

}  // namespace mllm::opencl
