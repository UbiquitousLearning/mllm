// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/GDNOp.hpp"
#include "mllm/backends/cpu/kernels/GDNKernel.hpp" // 引入我们之前写好的 Kernel

namespace mllm::cpu {

class CPUGDNOp final : public aops::GDNOp {
 public:
  // 构造函数直接透传 options
  explicit CPUGDNOp(const aops::GDNOpOptions& options) : aops::GDNOp(options) {}

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override {
    // 获取输入输出 Tensor
    const auto& s_prev = inputs[0];
    const auto& k = inputs[1];
    const auto& v = inputs[2];
    const auto& gate = inputs[3];
    const auto& beta = inputs[4];
    
    auto& s_new = outputs[0];
    int B = s_prev.size(0); // 获取 batch size

    // 调用我们写好的 GDNKernel 纯计算逻辑
    // 注意：这里使用 .ptr<float>() 获取底层数据指针
    GDNKernel::forward(
        s_prev.ptr<float>(), k.ptr<float>(), v.ptr<float>(),
        gate.ptr<float>(), beta.ptr<float>(),
        s_new.ptr<float>(),
        nullptr, nullptr, // output 和 q 暂时不用，传空指针
        B
    );
  }
};

// 工厂类，照着 GELU 的模板改的
class GDNOpFactory : public TypedOpFactory<OpTypes::kGDN, aops::GDNOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::GDNOpOptions& options) override {
    return std::make_shared<CPUGDNOp>(options);
  }
};

}  // namespace mllm::cpu
