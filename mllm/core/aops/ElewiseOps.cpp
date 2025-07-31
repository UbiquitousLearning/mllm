/**
 * @file ElewiseOps.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-31
 *
 */
#include "mllm/core/aops/ElewiseOps.hpp"

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

namespace mllm::aops {

__MLLM_ELEWISE_OP_IMPL(kAdd, AddOp);
__MLLM_ELEWISE_OP_IMPL(kSub, SubOp);
__MLLM_ELEWISE_OP_IMPL(kMul, MulOp);
__MLLM_ELEWISE_OP_IMPL(kDiv, DivOp);
__MLLM_ELEWISE_OP_IMPL(kNeg, NegOp);

}  // namespace mllm::aops

#undef __MLLM_ELEWISE_OP_IMPL