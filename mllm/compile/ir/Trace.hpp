/**
 * @file Trace.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-29
 *
 */
#pragma once

#include "mllm/compile/ir/Node.hpp"
#include "mllm/nn/Module.hpp"

namespace mllm::ir {

// The function below is for python binding
IRContext::ptr_t trace_(nn::Module& module, const std::vector<Tensor>& ref_inputs);

template<typename... Args>
IRContext::ptr_t trace(nn::Module& module, Args&&... args) {
  return trace_(module, std::vector<Tensor>{args...});
}

}  // namespace mllm::ir
