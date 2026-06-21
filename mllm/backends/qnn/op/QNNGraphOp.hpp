// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/GraphOps.hpp"

namespace mllm::qnn {

class QNNGraphBeginOp final : public aops::GraphBeginOp {
 public:
  explicit QNNGraphBeginOp(const aops::GraphBeginOpOptions& options);
};

class QNNGraphBeginOpFactory final : public TypedOpFactory<OpTypes::kGraphBegin, aops::GraphBeginOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GraphBeginOpOptions& options) override {
    return std::make_shared<QNNGraphBeginOp>(options);
  }
};

class QNNGraphEndOp final : public aops::GraphEndOp {
 public:
  explicit QNNGraphEndOp(const aops::GraphEndOpOptions& options);
};

class QNNGraphEndOpFactory final : public TypedOpFactory<OpTypes::kGraphEnd, aops::GraphEndOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GraphEndOpOptions& options) override {
    return std::make_shared<QNNGraphEndOp>(options);
  }
};

}  // namespace mllm::qnn
