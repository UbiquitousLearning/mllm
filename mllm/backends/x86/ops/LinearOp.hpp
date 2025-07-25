/**
 * @file LinearOp.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#pragma once

#include "mllm/core/aops/LinearOp.hpp"

namespace mllm::x86 {

class X86LinearOp final : public aops::LinearOp {
 public:
  explicit X86LinearOp(const aops::LinearOpOptions& options);

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
};

class X86LinearOpFactory : public TypedOpFactory<OpTypes::kLinear, aops::LinearOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::LinearOpOptions& options) override {
    return std::make_shared<X86LinearOp>(options);
  }
};

}  // namespace mllm::x86