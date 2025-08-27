// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_map>
#include <functional>
#include <vector>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/RTTIHelper.hpp"

namespace mllm::jit {

class JITRunnerStates {
  // TODO
};

//===----------------------------------------------------------------------===//
// How to perform each IR node
//===----------------------------------------------------------------------===//
template<typename __IR_NODE>
struct InstructionImpl {
  void instID() {}

  void execute(JITRunnerStates& states, const __IR_NODE::ptr_t& ir, std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               std::vector<ir::tensor::TensorValue::ptr_t>& outputs) {}
};

//===----------------------------------------------------------------------===//
// Runner
//===----------------------------------------------------------------------===//
class JITRunner {};

}  // namespace mllm::jit
