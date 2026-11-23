// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/LinearShadowCorrectionOp.hpp"

#include <array>
#include <string_view>

#include "mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/ShadowCorrectionPacketLayout.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/compile/ir/tensor/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::aops {

namespace {

constexpr std::string_view kShadowSuffix = ".shadow";

void registerParameter(ir::IRContext* ir_ctx, const Tensor& tensor) {
  if (!tensor || ir_ctx->lookupSymbolTable(tensor.name())) { return; }
  ir::IRWriterGuard guard(ir_ctx, ir_ctx->lookupSymbolTable("init")->cast_<ir::graph::SubGraphOp>()->getTopRegion());
  ir_ctx->create<ir::tensor::RegisterOp>(ir_ctx->create<ir::tensor::TensorValue>(tensor));
}

}  // namespace

LinearShadowCorrectionOp::LinearShadowCorrectionOp(const LinearShadowCorrectionOpOptions& options)
    : BaseOp(OpTypes::kLinearShadowCorrection), options_(options) {}

std::string LinearShadowCorrectionOp::parameterPrefix() const {
  const auto name = getName();
  MLLM_RT_ASSERT(name.size() > kShadowSuffix.size());
  MLLM_RT_ASSERT(name.compare(name.size() - kShadowSuffix.size(), kShadowSuffix.size(), kShadowSuffix) == 0);
  return name.substr(0, name.size() - kShadowSuffix.size());
}

void LinearShadowCorrectionOp::load(const ParameterFile::ptr_t& ploader) {
  const auto prefix = parameterPrefix();
  weight_ = ploader->pull(prefix + ".weight");
  weight_scale_ = ploader->pull(prefix + ".weight.scale");
  input_scale_ = ploader->pull(prefix + ".input_scale");
  output_scale_ = ploader->pull(prefix + ".output_scale");
  clip_input_ = ploader->pull(prefix + ".clip_input");
  clip_output_ = ploader->pull(prefix + ".clip_output");

  if (ploader->version() == ModelFileVersion::kV1) { weight_ = weight_.view({options_.in_channels, options_.out_channels}); }
  MLLM_RT_ASSERT_EQ(weight_.numel(), options_.in_channels * options_.out_channels);
  MLLM_RT_ASSERT_EQ(weight_.dtype(), kInt8);
  MLLM_RT_ASSERT_EQ(weight_scale_.numel(), 1);
  MLLM_RT_ASSERT_EQ(input_scale_.numel(), 1);
  MLLM_RT_ASSERT_EQ(output_scale_.numel(), 1);
  MLLM_RT_ASSERT_EQ(clip_input_.numel(), 1);
  MLLM_RT_ASSERT_EQ(clip_output_.numel(), 1);
}

void LinearShadowCorrectionOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto* ir_ctx = static_cast<ir::IRContext*>(trace_context);
  for (const auto& tensor :
       std::array<Tensor, 6>{weight_, weight_scale_, input_scale_, output_scale_, clip_input_, clip_output_}) {
    registerParameter(ir_ctx, tensor);
  }
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::LinearShadowCorrectionOp>(shared_from_this(), i_irs, o_irs);
}

void LinearShadowCorrectionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  NYI("LinearShadowCorrectionOp::forward not implemented in aops base.");
}

void LinearShadowCorrectionOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_RT_ASSERT_EQ(inputs.size(), 3);
  MLLM_RT_ASSERT(options_.in_channels > 0 && options_.out_channels > 0);
  MLLM_RT_ASSERT_EQ(options_.activation_bits, 16);

  const auto& original_input = inputs[0];
  const auto& quantized_output = inputs[1];
  const auto& baseline = inputs[2];
  MLLM_RT_ASSERT(!original_input.shape().empty());
  MLLM_RT_ASSERT(!quantized_output.shape().empty());
  MLLM_RT_ASSERT(!baseline.shape().empty());
  MLLM_RT_ASSERT_EQ(quantized_output.shape().back(), options_.out_channels);
  MLLM_RT_ASSERT_EQ(baseline.shape().back(), options_.out_channels);
  MLLM_RT_ASSERT_EQ(quantized_output.dtype(), kInt16);
  MLLM_RT_ASSERT_EQ(baseline.dtype(), kFloat32);
  MLLM_RT_ASSERT_EQ(original_input.device(), kCPU);
  MLLM_RT_ASSERT_EQ(quantized_output.device(), kCPU);
  MLLM_RT_ASSERT_EQ(baseline.device(), kCPU);

  const auto tokens = baseline.numel() / options_.out_channels;
  if (original_input.dtype() == kInt32) {
    using namespace ::shadow_correction_packet;
    MLLM_RT_ASSERT_EQ(options_.in_channels, in_width);
    MLLM_RT_ASSERT_EQ(options_.out_channels, out_width);
    MLLM_RT_ASSERT(tokens > 0 && tokens % tile_rows == 0);
    MLLM_RT_ASSERT_EQ(original_input.shape().back(), words_per_row);
    MLLM_RT_ASSERT_EQ(original_input.numel(), tokens * words_per_row);
    MLLM_RT_ASSERT(original_input.isContiguous() && quantized_output.isContiguous() && baseline.isContiguous());
  } else {
    MLLM_RT_ASSERT_EQ(original_input.dtype(), kFloat32);
    MLLM_RT_ASSERT_EQ(original_input.shape().back(), options_.in_channels);
    MLLM_RT_ASSERT_EQ(original_input.numel(), tokens * options_.in_channels);
  }
  MLLM_RT_ASSERT_EQ(quantized_output.numel(), tokens * options_.out_channels);
  outputs.emplace_back(Tensor::empty(baseline.shape(), kFloat32, kCPU));
}

void LinearShadowCorrectionOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  BaseOp::setup(inputs, outputs);
}

ParameterFile::ptr_t LinearShadowCorrectionOp::getParams() {
  // These tensors are aliases of the source Linear's checkpoint parameters.
  // Returning them again would duplicate names when a Module seals params().
  return ParameterFile::create();
}

}  // namespace mllm::aops
