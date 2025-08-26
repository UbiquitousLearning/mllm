// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/aops/ReduceOps.hpp"

namespace mllm::cpu {

class CPUReduceMaxOp final : public aops::ReduceMaxOp {
 public:
  explicit CPUReduceMaxOp(const aops::ReduceMaxOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUReduceMaxOpFactory final : public TypedOpFactory<OpTypes::kReduceMax, aops::ReduceMaxOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ReduceMaxOpOptions& options) override {
    return std::make_shared<CPUReduceMaxOp>(options);
  }
};

class CPUReduceMinOp final : public aops::ReduceMinOp {
 public:
  explicit CPUReduceMinOp(const aops::ReduceMinOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUReduceMinOpFactory final : public TypedOpFactory<OpTypes::kReduceMin, aops::ReduceMinOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ReduceMinOpOptions& options) override {
    return std::make_shared<CPUReduceMinOp>(options);
  }
};

class CPUReduceSumOp final : public aops::ReduceSumOp {
 public:
  explicit CPUReduceSumOp(const aops::ReduceSumOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUReduceSumOpFactory final : public TypedOpFactory<OpTypes::kReduceSum, aops::ReduceSumOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ReduceSumOpOptions& options) override {
    return std::make_shared<CPUReduceSumOp>(options);
  }
};

class CPUMeanOp final : public aops::MeanOp {
 public:
  explicit CPUMeanOp(const aops::MeanOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUMeanOpFactory final : public TypedOpFactory<OpTypes::kMean, aops::MeanOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MeanOpOptions& options) override {
    return std::make_shared<CPUMeanOp>(options);
  }
};

}  // namespace mllm::cpu
