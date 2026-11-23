// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "CustomLayers.hpp"
#include <memory>
#include <bit>
#include <cmath>
#include <string_view>

#include "mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/ShadowCorrectionPacketLayout.hpp"

#include "mllm/backends/qnn/QNNUtils.hpp"
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

ShadowCorrectionPacket::ShadowCorrectionPacket()
    : Layer(OpTypes::kDynamicOp_Start, ShadowCorrectionPacketOptions{}) {
  impl()->__forceSetOpType(static_cast<OpTypes>(Context::instance().lookupCustomizedOpId(kQNN, "ShadowCorrectionPacket")));
  impl()->__forceSetDevice(kQNN);
}

}  // namespace mllm::nn::qnn

// -------------------- Custom QNN Ops --------------------
namespace mllm::qnn {
void DequantizeAddOp::load(const mllm::ParameterFile::ptr_t& ploader) {
  std::string parameter_prefix = getName();
  // find the ".dequantize" suffix and replace it with ".bias"
  const auto pos = parameter_prefix.rfind("dequantize");
  if (pos == std::string::npos || pos + 10 != parameter_prefix.size()) {
    MLLM_ERROR_EXIT(1, "DequantizeAdd op name must end in dequantize: {}", getName());
  }
  parameter_prefix.erase(pos, 10);
  const std::string weight_name = parameter_prefix + "bias";
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

namespace {

namespace packet = ::shadow_correction_packet;

bool readShadowFlag(const Tensor& tensor) {
  switch (tensor.dtype()) {
    case kBool:
    case kUInt8: return tensor.ptr<uint8_t>()[0] != 0;
    case kInt8: return tensor.ptr<int8_t>()[0] != 0;
    case kInt16: return tensor.ptr<int16_t>()[0] != 0;
    case kInt32: return tensor.ptr<int32_t>()[0] != 0;
    default: MLLM_ERROR_EXIT(1, "Unsupported shadow correction flag dtype");
  }
  return false;
}

// Search the ordered positive FP32 bit patterns for the first value whose
// rounded code exceeds the corresponding signed endpoint.
uint32_t firstClippedMagnitude(float step, float max_code) {
  uint32_t lo = 1, hi = 0x7f800000u;
  while (lo != hi) {
    const uint32_t middle = lo + (hi - lo) / 2;
    const float value = std::bit_cast<float>(middle);
    if (std::round(value / step) > max_code) { hi = middle; }
    else { lo = middle + 1; }
  }
  MLLM_RT_ASSERT(lo < 0x7f800000u);
  return lo;
}

}  // namespace

void ShadowCorrectionPacketOp::load(const ParameterFile::ptr_t& file) {
  const auto name = getName();
  constexpr std::string_view suffix = ".packet";
  MLLM_RT_ASSERT(name.ends_with(suffix));
  const auto prefix = name.substr(0, name.size() - suffix.size());
  const auto scale = file->pull(prefix + ".input_scale");
  MLLM_RT_ASSERT_EQ(scale.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(scale.numel(), 1);
  const float step = scale.ptr<float>()[0] / 32767.0f;
  MLLM_RT_ASSERT(step > 0.0f && (std::bit_cast<uint32_t>(step) & 0x7f800000u) != 0x7f800000u);
  const auto clip_input = file->pull(prefix + ".clip_input");
  const auto clip_output = file->pull(prefix + ".clip_output");
  MLLM_RT_ASSERT_EQ(clip_input.numel(), 1);
  MLLM_RT_ASSERT_EQ(clip_output.numel(), 1);
  controls_ = {firstClippedMagnitude(step, 32767.0f), firstClippedMagnitude(step, 32768.0f),
               static_cast<uint32_t>((readShadowFlag(clip_input) ? packet::clip_input : 0)
                                   | (readShadowFlag(clip_output) ? packet::clip_output : 0))};
}

void ShadowCorrectionPacketOp::trace(void* trace_context, const std::vector<Tensor>& inputs,
                                     std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  ir_ctx->create<ir::linalg::CustomizedOp>(shared_from_this(), ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs),
                                          ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs));
}

void ShadowCorrectionPacketOp::forward(const std::vector<Tensor>&, std::vector<Tensor>&) {
  MLLM_ERROR_EXIT(1, "ShadowCorrectionPacket must execute in a finalized QNN graph");
}

void ShadowCorrectionPacketOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT(inputs.size() == 2 || inputs.size() == 3);
  const auto& activation = inputs[0];
  const auto& projection = inputs[1];
  MLLM_RT_ASSERT(activation.rank() == 4 && projection.rank() == 4);
  const auto& shape = activation.shape();
  const auto rows = shape[1] * shape[2];
  MLLM_RT_ASSERT(shape[0] == 1 && rows > 0 && rows % packet::tile_rows == 0);
  MLLM_RT_ASSERT(activation.device() == kQNN && activation.dtype() == kFloat32 && shape[3] == packet::in_width);
  MLLM_RT_ASSERT(projection.device() == kQNN && projection.dtype() == kInt16);
  MLLM_RT_ASSERT_EQ(projection.shape(), (Tensor::shape_t{1, shape[1], shape[2], packet::out_width}));
  if (inputs.size() == 3) {
    MLLM_RT_ASSERT(inputs[2].device() == kQNN && inputs[2].dtype() == kInt16);
    MLLM_RT_ASSERT_EQ(inputs[2].shape(), shape);
  }
  outputs.emplace_back(Tensor::empty({1, rows, 1, packet::words_per_row}, kInt32, kQNN));
}

}  // namespace mllm::qnn
