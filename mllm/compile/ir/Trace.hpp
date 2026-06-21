// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/compile/ir/Node.hpp"
#include "mllm/nn/Module.hpp"

namespace mllm::ir {

// The function below is for python binding
IRContext::ptr_t trace_(nn::Module& module, const std::vector<Tensor>& ref_inputs, const std::vector<AnyValue>& args = {});

template<typename... Args>
IRContext::ptr_t trace(nn::Module& module, Args&&... args) {
  std::vector<Tensor> tensors;
  std::vector<AnyValue> others;
  (..., [&] {
    // The type must can be inference in compile time
    using CleanType = std::decay_t<decltype(args)>;
    if constexpr (std::is_convertible_v<CleanType, Tensor>) {
      tensors.push_back(std::forward<Args>(args));
    } else if constexpr (std::is_convertible_v<CleanType, AnyValue>) {
      others.push_back(std::forward<Args>(args));
    } else {
      static_assert(always_false<CleanType>::value, "Unsupported argument type!");
    }
  }());
  return trace_(module, tensors, others);
}

namespace lowlevel {
void traceExternalValue();

void traceComment(const std::string& comment);

void traceStart();

IRContext::ptr_t traceStop();

template<typename... Args>
std::vector<Tensor> traceModule(nn::Module& module, Args&&... args) {
  std::vector<Tensor> tensors;
  std::vector<AnyValue> others;
  (..., [&] {
    // The type must can be inference in compile time
    using CleanType = std::decay_t<decltype(args)>;
    if constexpr (std::is_convertible_v<CleanType, Tensor>) {
      tensors.push_back(std::forward<Args>(args));
    } else if constexpr (std::is_convertible_v<CleanType, AnyValue>) {
      others.push_back(std::forward<Args>(args));
    } else {
      static_assert(always_false<CleanType>::value, "Unsupported argument type!");
    }
  }());
  // Forward
  return module.__trace(tensors, others);
}

void traceYield();

void traceContinue();

}  // namespace lowlevel
}  // namespace mllm::ir
