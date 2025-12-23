// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/OpTypes.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/backends/qnn/aot/visitor/Base.hpp"

namespace mllm::qnn::aot {

class QnnAOTAddPattern : public QnnAOTBasePattern {
 public:
  bool isMatch(const mllm::ir::op_ptr_t& op) override;

  bool addNode(const std::string& g_name, const ir::op_ptr_t& op, const std::vector<ir::tensor::TensorValue::ptr_t>& inputs,
               const std::vector<ir::tensor::TensorValue::ptr_t>& outputs) override;

  static inline std::pair<OpTypes, std::shared_ptr<QnnAOTAddPattern>> create() {
    return {OpTypes::kAdd, std::make_shared<QnnAOTAddPattern>()};
  }
};

}  // namespace mllm::qnn::aot
