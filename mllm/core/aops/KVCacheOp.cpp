// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/KVCacheOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

KVCacheOp::KVCacheOp(const KVCacheOpOptions& options) : BaseOp(OpTypes::kKVCache), options_(options) {}

void KVCacheOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void KVCacheOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::KVCacheOp>(shared_from_this(), i_irs, o_irs);
}

void KVCacheOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("KVCacheOp::forward not implemented in aops base.");
}

void KVCacheOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Input is always [B, H, S, D]
  const int B = inputs[0].shape()[0];
  const int S = inputs[0].shape()[2];
  const int D = inputs[0].shape()[3];
  const DataTypes dtype = inputs[0].dtype();

  // inputs[0] is k tensor, inputs[1] is v tensor
  // outputs[0] is updated k tensor, outputs[1] is updated v tensor
  outputs.emplace_back(Tensor::empty({B, options_.kv_head, S, D}));
  outputs.emplace_back(Tensor::empty({B, options_.kv_head, S, D}));
}

void KVCacheOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void KVCacheOp::setLayerIndex(int32_t layer_idx) { options_.layer_idx = layer_idx; }

void KVCacheOp::clearCache() { NYI("KVCacheOp::clearCache not implemented in aops base."); }

}  // namespace mllm::aops
