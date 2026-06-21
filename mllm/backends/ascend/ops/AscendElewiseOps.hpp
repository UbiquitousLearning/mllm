// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/OpTypes.hpp"

namespace mllm::ascend {

class AscendAddOp final : public aops::AddOp {
 public:
  explicit AscendAddOp(const aops::AddOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendAddOpFactory final : public TypedOpFactory<OpTypes::kAdd, aops::AddOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::AddOpOptions& options) override {
    return std::make_shared<AscendAddOp>(options);
  }
};

class AscendSubOp final : public aops::SubOp {
 public:
  explicit AscendSubOp(const aops::SubOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendSubOpFactory final : public TypedOpFactory<OpTypes::kSub, aops::SubOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SubOpOptions& options) override {
    return std::make_shared<AscendSubOp>(options);
  }
};

class AscendMulOp final : public aops::MulOp {
 public:
  explicit AscendMulOp(const aops::MulOpOptions& options);

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class AscendMulOpFactory final : public TypedOpFactory<OpTypes::kMul, aops::MulOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MulOpOptions& options) override {
    return std::make_shared<AscendMulOp>(options);
  }
};

}  // namespace mllm::ascend