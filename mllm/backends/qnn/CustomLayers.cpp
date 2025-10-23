// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "CustomLayers.hpp"
#include <memory>
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/mllm.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

// -------------------- Custom QNN Layers --------------------
namespace mllm::nn::qnn {
DequantizeAdd::DequantizeAdd()
    : Layer(OpTypes::kDynamicOp_Start, DequantizeAddOpOptions{.dtype = kFloat32, .out_channels = 0}) {
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kQNN, "DequantizeAdd"));
  this->impl()->__forceSetDevice(kQNN);
}

DequantizeAdd::DequantizeAdd(DataTypes dtype, int32_t out_channels)
    : Layer(OpTypes::kDynamicOp_Start, DequantizeAddOpOptions{.dtype = dtype, .out_channels = out_channels}) {
  this->impl()->__forceSetOpType((mllm::OpTypes)mllm::Context::instance().lookupCustomizedOpId(mllm::kQNN, "DequantizeAdd"));
  this->impl()->__forceSetDevice(kQNN);
}

}  // namespace mllm::nn::qnn

// -------------------- Custom QNN Ops --------------------
namespace mllm::qnn {
void DequantizeAddOp::load(const mllm::ParameterFile::ptr_t& ploader) {
  std::string weight_name = getName();
  // find the ".dequantize" suffix and replace it with ".bias"
  auto pos = weight_name.find("dequantize");
  if (pos != -1) { weight_name.erase(pos, 10); }
  weight_name += "bias";

  switch (ploader->version()) {
    case ModelFileVersion::kV1: {
      weight_ = ploader->pull(weight_name);
      weight_ = weight_.view({1, 1, 1, options_.out_channels});
      break;
    }
    case ModelFileVersion::kUserTemporary:
    case ModelFileVersion::kV2: {
      weight_ = ploader->pull(weight_name);
      weight_ = weight_.view({1, 1, 1, options_.out_channels});
      break;
    }
    default: NYI("Unsupported model file version")
  }
}

void DequantizeAddOp::trace(void* trace_context, const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::CustomizedOp>(shared_from_this(), i_irs, o_irs);
}
void DequantizeAddOp::reshape(const std::vector<mllm::Tensor>& inputs, std::vector<mllm::Tensor>& outputs) {
  // CastType operation maintains the same shape
  assert(inputs.size() == 1);
  const auto& input = inputs[0];

  outputs.emplace_back(Tensor::empty(input.shape(), options_.dtype, input.device()));
}

}  // namespace mllm::qnn
