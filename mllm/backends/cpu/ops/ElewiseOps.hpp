// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"

namespace mllm::cpu {

class CPUAddOp final : public aops::AddOp {
 public:
  explicit CPUAddOp(const aops::AddOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUSubOp final : public aops::SubOp {
 public:
  explicit CPUSubOp(const aops::SubOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUMulOp final : public aops::MulOp {
 public:
  explicit CPUMulOp(const aops::MulOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUDivOp final : public aops::DivOp {
 public:
  explicit CPUDivOp(const aops::DivOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUNegOp final : public aops::NegOp {
 public:
  explicit CPUNegOp(const aops::NegOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUAbsOp final : public aops::AbsOp {
 public:
  explicit CPUAbsOp(const aops::AbsOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUClipOp final : public aops::ClipOp {
 public:
  explicit CPUClipOp(const aops::ClipOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPUAddOpFactory final : public TypedOpFactory<OpTypes::kAdd, aops::AddOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::AddOpOptions& options) override {
    return std::make_shared<CPUAddOp>(options);
  }
};

class CPUSubOpFactory : public TypedOpFactory<OpTypes::kSub, aops::SubOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SubOpOptions& options) override {
    return std::make_shared<CPUSubOp>(options);
  }
};

class CPUMulOpFactory : public TypedOpFactory<OpTypes::kMul, aops::MulOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MulOpOptions& options) override {
    return std::make_shared<CPUMulOp>(options);
  }
};

class CPUDivOpFactory : public TypedOpFactory<OpTypes::kDiv, aops::DivOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::DivOpOptions& options) override {
    return std::make_shared<CPUDivOp>(options);
  }
};

class CPUNegOpFactory : public TypedOpFactory<OpTypes::kNeg, aops::NegOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::NegOpOptions& options) override {
    return std::make_shared<CPUNegOp>(options);
  }
};

class CPUAbsOpFactory : public TypedOpFactory<OpTypes::kAbs, aops::AbsOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::AbsOpOptions& options) override {
    return std::make_shared<CPUAbsOp>(options);
  }
};

class CPULogOp : public aops::LogOp {
 public:
  explicit CPULogOp(const aops::LogOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class CPULogOpFactory : public TypedOpFactory<OpTypes::kLog, aops::LogOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LogOpOptions& options) override {
    return std::make_shared<CPULogOp>(options);
  }
};

class CPUClipOpFactory : public TypedOpFactory<OpTypes::kClip, aops::ClipOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::ClipOpOptions& options) override {
    return std::make_shared<CPUClipOp>(options);
  }
};

}  // namespace mllm::cpu
