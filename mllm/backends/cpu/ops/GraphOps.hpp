// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/GraphOps.hpp"

namespace mllm::cpu {

class CPUGraphBeginOp final : public aops::GraphBeginOp {
 public:
  explicit CPUGraphBeginOp(const aops::GraphBeginOpOptions& options);
};

class CPUGraphBeginOpFactory final : public TypedOpFactory<OpTypes::kGraphBegin, aops::GraphBeginOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GraphBeginOpOptions& options) override {
    return std::make_shared<CPUGraphBeginOp>(options);
  }
};

class CPUGraphEndOp final : public aops::GraphEndOp {
 public:
  explicit CPUGraphEndOp(const aops::GraphEndOpOptions& options);
};

class CPUGraphEndOpFactory final : public TypedOpFactory<OpTypes::kGraphEnd, aops::GraphEndOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GraphEndOpOptions& options) override {
    return std::make_shared<CPUGraphEndOp>(options);
  }
};

}  // namespace mllm::cpu
