// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/Conv1DOp.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPUConv1DOp::CPUConv1DOp(const aops::Conv1DOpOptions& options) : aops::Conv1DOp(options) {}

void CPUConv1DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto i_shape = input.shape();
  auto o_shape = output.shape();

  // input shape: [batch, in_channels, sequence]
  // output shape: [batch, out_channels, out_sequence]
  int batch = i_shape[0];
  int in_channels = i_shape[1];
  int sequence = i_shape[2];

  int out_channels = o_shape[1];
  int out_sequence = o_shape[2];

  int kernel_size = options_.kernel_size;
  int stride = options_.stride;
  int padding = options_.padding;

  auto input_ptr = input.ptr<float>();
  MLLM_RT_ASSERT(weight_.dtype() == kFloat32);
  auto weight_ptr = weight_.ptr<float>();
  auto output_ptr = output.ptr<float>();

  float* bias_ptr = nullptr;
  if (options_.bias && !bias_.isNil()) { bias_ptr = bias_.ptr<float>(); }

  std::fill_n(output_ptr, output.numel(), 0.0f);

  int total_iterations = batch * out_channels * out_sequence;

  switch (output.dtype()) {
    case kFloat32:
      MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, 4, idx, 0, total_iterations, 1, {
        // 从一维索引还原三维索引
        int b = idx / (out_channels * out_sequence);
        int oc = (idx % (out_channels * out_sequence)) / out_sequence;
        int os = idx % out_sequence;

        float sum = 0.0f;

        for (int ic = 0; ic < in_channels; ic++) {
          for (int k = 0; k < kernel_size; k++) {
            int input_pos = os * stride - padding + k;

            if (input_pos >= 0 && input_pos < sequence) {
              // input index: [batch, in_channel, sequence]
              int input_idx = b * (in_channels * sequence) + ic * sequence + input_pos;

              // weight index: [out_channel, in_channel, kernel_size]
              int weight_idx = oc * (in_channels * kernel_size) + ic * kernel_size + k;

              sum += input_ptr[input_idx] * weight_ptr[weight_idx];
            }
          }
        }

        if (bias_ptr) { sum += bias_ptr[oc]; }

        // output index: [batch, out_channel, out_sequence]
        int output_idx = b * (out_channels * out_sequence) + oc * out_sequence + os;
        output_ptr[output_idx] = sum;
      });
      break;
    default: NYI("Conv1D: unsupported data type");
  }
}

}  // namespace mllm::cpu