// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/ISTFTOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::aops {

ISTFTOp::ISTFTOp(const ISTFTOpOptions& options) : BaseOp(OpTypes::kISTFT), options_(options) {}

void ISTFTOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ISTFTOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::ISTFTOp>(shared_from_this(), i_irs, o_irs);
}

void ISTFTOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void ISTFTOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // options validate
  MLLM_RT_ASSERT(options_.n_fft > 0);  // n_fft is required
  MLLM_RT_ASSERT(options_.onesided);   // onesided only supports true
  if (options_.hop_length == 0) {
    options_.hop_length = options_.n_fft / 4;  // default hop_length
  }
  if (options_.win_length == 0) {
    options_.win_length = options_.n_fft;  // default win_length
  } else if (options_.win_length > options_.n_fft) {
    MLLM_WARN("ISTFT: win_length ({}) > n_fft ({}), clipping to n_fft", options_.win_length, options_.n_fft);
    options_.win_length = options_.n_fft;
  }

  MLLM_RT_ASSERT(inputs[0].shape().size() == 3 || inputs[0].shape().size() == 4);
  MLLM_RT_ASSERT(inputs.size() > 1 && inputs[1].shape().back() == options_.win_length);

  if (options_.center && options_.pad_mode != "reflect" && options_.pad_mode != "constant") {
    MLLM_WARN("ISTFT: center=True requires pad_mode to be 'reflect' or 'constant', got '{}'. set pad_mode to 'reflect'",
              options_.pad_mode);
    options_.pad_mode = "reflect";
  }

  auto& input = inputs[0];
  auto& output = outputs[0];

  // Get input dimensions
  auto input_shape = input.shape();
  int batch_size = input_shape[0];
  int freq_bins = input_shape[1];
  int n_frames = input_shape[2];

  // ISTFT parameters
  int n_fft = options_.n_fft;
  int hop_length = options_.hop_length;
  bool center = options_.center;

  // Calculate output dimensions
  int signal_length = (n_frames - 1) * hop_length + n_fft;
  options_.length = signal_length;
  if (center) {
    signal_length = signal_length - 2 * (n_fft / 2);
  } else if (options_.pad_mode == "same") {
    // center==false，remove padding of (win_length - hop_length) // 2
    int padding = (options_.win_length - hop_length) / 2;
    signal_length -= 2 * padding;
  }

  // Output shape: [batch_size, signal_length] with float dtype
  outputs.emplace_back(Tensor::empty({batch_size, signal_length}, kFloat32, input.device()));
}

void ISTFTOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops