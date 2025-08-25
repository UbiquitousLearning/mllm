// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/DataTypes.hpp"

#define __MLLM_ELEWISE_OP_IMPL(types, name)                                                                \
  name::name(const name##Options& options) : BaseOp(OpTypes::types), options_(options) {}                  \
  void name::load(const ParameterFile::ptr_t& ploader) {}                                                  \
  void name::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { \
    auto ctx = (ir::IRContext*)trace_context;                                                              \
    auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);                                            \
    auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);                                           \
    ctx->create<ir::linalg::name>(shared_from_this(), i_irs, o_irs);                                       \
  }                                                                                                        \
  void name::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {                    \
    MLLM_WARN(#name "::forward is not implemented");                                                       \
  }                                                                                                        \
  void name::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {                    \
    Tensor output_0 = Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device());             \
    outputs.emplace_back(output_0);                                                                        \
  }                                                                                                        \
  void name::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

// for unary ops, reshape don't consider shape broadcast, and dtype of output needs to be handled for Abs op
#define __MLLM_ELEWISE_UNARY_OP_IMPL(types, name)                                                          \
  name::name(const name##Options& options) : BaseOp(OpTypes::types), options_(options) {}                  \
  void name::load(const ParameterFile::ptr_t& ploader) {}                                                  \
  void name::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { \
    auto ctx = (ir::IRContext*)trace_context;                                                              \
    auto i_irs = ir::tensor::wrapTensors2TensorIR(ctx, inputs);                                            \
    auto o_irs = ir::tensor::wrapTensors2TensorIR(ctx, outputs);                                           \
    ctx->create<ir::linalg::name>(shared_from_this(), i_irs, o_irs);                                       \
  }                                                                                                        \
  void name::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {                    \
    MLLM_WARN(#name "::forward is not implemented");                                                       \
  }                                                                                                        \
  void name::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {                    \
    /* NOTE: currently only takes into Abs op, other ops may still have complex outputs */                 \
    if (inputs[0].dtype() == kComplexFloat32 || inputs[0].dtype() == kComplexFloat64) {                    \
      Tensor output_0 = Tensor::empty(inputs[0].shape(), kFloat32, inputs[0].device());                    \
      outputs.emplace_back(output_0);                                                                      \
      return;                                                                                              \
    } else {                                                                                               \
      Tensor output_0 = Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device());           \
      outputs.emplace_back(output_0);                                                                      \
    }                                                                                                      \
  }                                                                                                        \
  void name::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

namespace mllm::aops {

__MLLM_ELEWISE_OP_IMPL(kAdd, AddOp);
__MLLM_ELEWISE_OP_IMPL(kSub, SubOp);
__MLLM_ELEWISE_OP_IMPL(kMul, MulOp);
__MLLM_ELEWISE_OP_IMPL(kDiv, DivOp);
__MLLM_ELEWISE_OP_IMPL(kNeg, NegOp);

// ---------- Unary Ops
__MLLM_ELEWISE_UNARY_OP_IMPL(kAbs, AbsOp);

}  // namespace mllm::aops

#undef __MLLM_ELEWISE_OP_IMPL