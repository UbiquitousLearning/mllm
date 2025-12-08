// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/backends/opencl/runtime/OpenCLRuntime.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include <vector>

namespace mllm::opencl {
class OpenCLAddOp final : public aops::AddOp {
 public:
  explicit OpenCLAddOp(const aops::AddOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
};

class OpenCLAddOpFactory : public TypedOpFactory<OpTypes::kAdd, aops::AddOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::AddOpOptions& options) override {
    return std::make_shared<OpenCLAddOp>(options);
  }
};

class OpenCLSubOp final : public aops::SubOp {
 public:
  explicit OpenCLSubOp(const aops::SubOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
};

class OpenCLSubOpFactory : public TypedOpFactory<OpTypes::kSub, aops::SubOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::SubOpOptions& options) override {
    return std::make_shared<OpenCLSubOp>(options);
  }
};

class OpenCLMulOp final : public aops::MulOp {
 public:
  explicit OpenCLMulOp(const aops::MulOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
};

class OpenCLMulOpFactory : public TypedOpFactory<OpTypes::kMul, aops::MulOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::MulOpOptions& options) override {
    return std::make_shared<OpenCLMulOp>(options);
  }
};

class OpenCLDivOp final : public aops::DivOp {
 public:
  explicit OpenCLDivOp(const aops::DivOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 private:
  std::shared_ptr<KernelWrap> kernel_fp32_buffer_ = nullptr;
};

class OpenCLDivOpFactory : public TypedOpFactory<OpTypes::kDiv, aops::DivOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::DivOpOptions& options) override {
    return std::make_shared<OpenCLDivOp>(options);
  }
};

}  // namespace mllm::opencl