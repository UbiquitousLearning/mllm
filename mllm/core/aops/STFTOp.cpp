// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/STFTOp.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"
#include "mllm/utils/Log.hpp"

namespace mllm::aops {

STFTOp::STFTOp(const STFTOpOptions& options) : BaseOp(OpTypes::kSTFT), options_(options) {}

void STFTOp::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void STFTOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto ir_ctx = (ir::IRContext*)trace_context;
  auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
  auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
  ir_ctx->create<ir::linalg::STFTOp>(shared_from_this(), i_irs, o_irs);
}

void STFTOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void STFTOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // options validate
  MLLM_RT_ASSERT(options_.n_fft > 0);  // n_fft is required
  MLLM_RT_ASSERT(options_.onesided);   // onesided only supports true
  if (options_.hop_length == 0) {
    options_.hop_length = options_.n_fft / 4;  // default hop_length
  }
  if (options_.win_length == 0) {
    options_.win_length = options_.n_fft;  // default win_length
  } else if (options_.win_length > options_.n_fft) {
    MLLM_WARN("STFT: win_length ({}) > n_fft ({}), clipping to n_fft", options_.win_length, options_.n_fft);
    options_.win_length = options_.n_fft;
  }
  MLLM_RT_ASSERT(inputs[0].shape().size() == 1 || inputs[0].shape().size() == 2);
  MLLM_RT_ASSERT(inputs.size() > 1 && inputs[1].shape().back() == options_.win_length);

  if (options_.center && options_.pad_mode != "reflect" && options_.pad_mode != "constant") {
    MLLM_WARN("STFT: center=True requires pad_mode to be 'reflect' or 'constant', got '{}'. set pad_mode to 'reflect'",
              options_.pad_mode);
    options_.pad_mode = "reflect";
  }

  auto& input = inputs[0];
  auto& output = outputs[0];

  // Get input dimensions
  auto input_shape = input.shape();
  int batch_size = input_shape.size() == 1 ? 1 : input_shape[0];
  int signal_length = input_shape.size() == 1 ? input_shape[0] : input_shape[1];

  // STFT parameters
  int n_fft = options_.n_fft;
  int hop_length = options_.hop_length;
  int win_length = options_.win_length;
  bool center = options_.center;

  // If center=true, pad signal with n_fft/2 on both sides
  if (center) { signal_length += 2 * (n_fft / 2); }

  // Calculate output dimensions
  int n_frames = 1 + (signal_length - win_length) / hop_length;

  int freq_bins = options_.onesided ? n_fft / 2 + 1 : n_fft;

  if (options_.return_complex) {
    // Output shape: [batch_size, freq_bins, n_frames] with complex dtype
    outputs.emplace_back(Tensor::empty({batch_size, freq_bins, n_frames},
                                       input.dtype() == kFloat32 ? kComplexFloat32 : kComplexFloat64, input.device()));
  } else {
    // Output shape: [batch_size, freq_bins, n_frames, 2] with real dtype
    outputs.emplace_back(Tensor::empty({batch_size, freq_bins, n_frames, 2}, input.dtype(), input.device()));
  }
}

void STFTOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { BaseOp::setup(inputs, outputs); }

}  // namespace mllm::aops