// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/Tensor.hpp"
#include "mllm/compile/ir/builtin/Interface.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/NodeRTTIClassOfImpl.hpp"

namespace mllm::ir::tensor {

class TensorIRValue : public Val {
 public:
  DEFINE_SPECIFIC_IR_CLASS(TensorIRValue);

  ~TensorIRValue() override;

  TensorIRValue();

  explicit TensorIRValue(NodeKind kind);

  static inline bool classof(const Node* node) { RTTI_RK_VAL_TENSORIRVAL_IMPL(node); }
};

class TensorValue : public TensorIRValue, public SymbolInterface<TensorValue> {
 public:
  DEFINE_SPECIFIC_IR_CLASS(TensorValue);

  ~TensorValue() override;
  TensorValue();

  static ptr_t build(IRContext* ctx, const Tensor& tensor);

  static inline bool classof(const Node* node) { RTTI_RK_VAL_TENSORIRVAL_TENSORVAL_IMPL(node); }

  void dump(IRPrinter& p) override;

  Tensor tensor_;
};

static inline std::vector<TensorValue::ptr_t> wrapTensors2TensorIR(IRContext* ctx, const std::vector<Tensor>& tensors,
                                                                   bool no_memory_side_effect = false) {
  std::vector<TensorValue::ptr_t> tensor_ir_values;
  for (auto& t : tensors) {
    if (!no_memory_side_effect && ctx->isCacheInputOutputTensor(t.uuid())) {
      tensor_ir_values.emplace_back(ctx->getCacheInputOutputTensor(t.uuid())->cast_<ir::tensor::TensorValue>());
    } else {
      auto ret = ctx->create<TensorValue>(t);
      ctx->cacheInputOutputTensor(t.uuid(), ret);
      tensor_ir_values.emplace_back(ret);
    }
  }
  return tensor_ir_values;
}

}  // namespace mllm::ir::tensor
