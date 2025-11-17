// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/AvgPool1dOp.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPUAvgPool1dOp::CPUAvgPool1dOp(const aops::AvgPool1dOpOptions& options) : aops::AvgPool1dOp(options) {}

void CPUAvgPool1dOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto i_shape = input.shape();
  auto o_shape = output.shape();

  // input shape: [batch, channels, length]
  // output shape: [batch, channels, pooled_length]
  int batch = i_shape[0];
  int channels = i_shape[1];
  int length = i_shape[2];

  int pooled_length = o_shape[2];

  int kernel_size = options_.kernel_size;
  int stride = options_.stride;
  int padding = options_.padding;

  auto input_ptr = input.ptr<float>();
  auto output_ptr = output.ptr<float>();

  std::fill_n(output_ptr, output.numel(), 0.0f);

  int total_iterations = batch * channels * pooled_length;

  switch (output.dtype()) {
    case kFloat32:
      MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, 4, idx, 0, total_iterations, 1, {
        // Decode 1D index to 3D indices
        int b = idx / (channels * pooled_length);
        int c = (idx % (channels * pooled_length)) / pooled_length;
        int out_pos = idx % pooled_length;

        float sum = 0.0f;
        int count = 0;

        // Calculate the range of input positions for this output position
        int start_pos = out_pos * stride - padding;

        for (int k = 0; k < kernel_size; ++k) {
          int input_pos = start_pos + k;

          // Check if input_pos is within bounds
          if (input_pos >= 0 && input_pos < length) {
            // input index: [batch, channel, length]
            int input_idx = b * (channels * length) + c * length + input_pos;
            sum += input_ptr[input_idx];
            count++;
          } else if (options_.count_include_pad) {
            // If count_include_pad is true, count padded positions
            count++;
          }
        }

        // Calculate average
        // output index: [batch, channel, pooled_length]
        int output_idx = b * (channels * pooled_length) + c * pooled_length + out_pos;
        if (count > 0) {
          output_ptr[output_idx] = sum / static_cast<float>(count);
        }
      });
      break;
    default: NYI("AvgPool1d: unsupported data type");
  }
}

}  // namespace mllm::cpu
