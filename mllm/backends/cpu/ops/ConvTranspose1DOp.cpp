// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/ConvTranspose1DOp.hpp"
#include "mllm/core/Parallel.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

CPUConvTranspose1DOp::CPUConvTranspose1DOp(const aops::ConvTranspose1DOpOptions& options)
    : aops::ConvTranspose1DOp(options) {}

void CPUConvTranspose1DOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& input = inputs[0];
  auto& output = outputs[0];

  auto i_shape = input.shape();
  auto o_shape = output.shape();

  // input shape: [batch, in_channels, sequence]
  // output shape: [batch, out_channels, out_sequence]
  const int batch = i_shape[0];
  const int in_channels = i_shape[1];
  const int sequence = i_shape[2];

  const int out_channels = o_shape[1];
  const int out_sequence = o_shape[2];

  const int kernel_size = options_.kernel_size;
  const int stride = options_.stride;
  const int padding = options_.padding;
  const int dilation = options_.dilation;
  const int groups = options_.groups;

  const int in_channels_per_group = in_channels / groups;
  const int out_channels_per_group = out_channels / groups;

  MLLM_RT_ASSERT(weight_.dtype() == kFloat32);
  const auto* weight_ptr = weight_.ptr<float>();
  const auto* input_ptr = input.ptr<float>();
  auto* output_ptr = output.ptr<float>();

  float* bias_ptr = nullptr;
  if (options_.bias && !bias_.isNil()) { bias_ptr = bias_.ptr<float>(); }

  std::fill_n(output_ptr, output.numel(), 0.0f);

  const int total_iterations = batch * out_channels * out_sequence;

  switch (output.dtype()) {
    case kFloat32:
      MLLM_CONDITIONAL_PARALLEL_FOR(options_.getThreads() > 1, 4, idx, 0, total_iterations, 1, {
        int b = idx / (out_channels * out_sequence);
        int oc = (idx % (out_channels * out_sequence)) / out_sequence;
        int out_pos = idx % out_sequence;

        const int group_idx = oc / out_channels_per_group;
        const int oc_in_group = oc % out_channels_per_group;

        float sum = 0.0f;

        for (int ic_in_group = 0; ic_in_group < in_channels_per_group; ++ic_in_group) {
          const int ic = group_idx * in_channels_per_group + ic_in_group;
          const int base_input_idx = b * (in_channels * sequence) + ic * sequence;

          const int base_weight_idx = (ic * out_channels_per_group + oc_in_group) * kernel_size;

          for (int k = 0; k < kernel_size; ++k) {
            int input_pos = out_pos + padding - k * dilation;
            if (input_pos % stride != 0) { continue; }
            input_pos /= stride;
            if (input_pos < 0 || input_pos >= sequence) { continue; }

            const int input_idx = base_input_idx + input_pos;
            const int weight_idx = base_weight_idx + k;

            sum += input_ptr[input_idx] * weight_ptr[weight_idx];
          }
        }

        if (bias_ptr) { sum += bias_ptr[oc]; }

        const int output_idx = b * (out_channels * out_sequence) + oc * out_sequence + out_pos;
        output_ptr[output_idx] = sum;
      });
      break;
    default: NYI("ConvTranspose1D: unsupported data type");
  }
}

}  // namespace mllm::cpu
