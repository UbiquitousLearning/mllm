// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/FlashAttention2Op.hpp"
#include <stdexcept>
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

FlashAttention2Op::FlashAttention2Op(const FlashAttention2OpOptions& options)
    : BaseOp(OpTypes::kFlashAttention2), options_(options) {}

void FlashAttention2Op::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void FlashAttention2Op::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::FlashAttention2Op>(shared_from_this(), i_irs, o_irs);
}

void FlashAttention2Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("FlashAttention2Op::forward not implemented in aops base.");
}

void FlashAttention2Op::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  if (inputs.size() != 3) throw std::invalid_argument("FlashAttention2 expects Q, K and V");
  if (options_.layout != FlashAttention2Layout::kBSHD && options_.layout != FlashAttention2Layout::kBHSD) {
    throw std::invalid_argument("FlashAttention2 received an unknown layout");
  }
  if (options_.sliding_window < 0 || (!options_.causal_mask && options_.sliding_window != 0)) {
    throw std::invalid_argument("FlashAttention2 sliding windows require causal attention and a positive width");
  }
  for (const auto& tensor : inputs) {
    if (tensor.isNil() || tensor.shape().size() != 4) {
      throw std::invalid_argument("FlashAttention2 expects rank-4 Q, K and V");
    }
    for (int32_t size : tensor.shape()) {
      if (size <= 0) throw std::invalid_argument("FlashAttention2 extents must be positive");
    }
    if (tensor.dtype() != inputs[0].dtype() || tensor.device() != inputs[0].device()) {
      throw std::invalid_argument("FlashAttention2 requires one dtype and device for all inputs");
    }
  }
  const int h = options_.layout == FlashAttention2Layout::kBSHD ? 2 : 1;
  const int s = 3 - h;
  const auto q = inputs[0].shape(), k = inputs[1].shape(), v = inputs[2].shape();
  if (q[0] != k[0] || q[0] != v[0] || q[3] != k[3] || q[h] % k[h] != 0 || k[h] != v[h] || k[s] != v[s]
      || (options_.causal_mask && q[s] > k[s])) {
    throw std::invalid_argument("FlashAttention2 incompatible batch, heads, dimensions, or causal suffix");
  }
  if ((options_.B != 0 && options_.B != q[0]) || (options_.q_head != 0 && options_.q_head != q[h])
      || (options_.kv_head != 0 && options_.kv_head != k[h]) || (options_.D != 0 && options_.D != q[3])) {
    throw std::invalid_argument("FlashAttention2 option dimensions disagree with inputs");
  }
  auto output_shape = q;
  output_shape[3] = v[3];
  outputs.emplace_back(Tensor::empty(output_shape, inputs[0].dtype(), inputs[0].device()));
}

void FlashAttention2Op::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

}  // namespace mllm::aops
