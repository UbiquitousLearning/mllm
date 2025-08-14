// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/MultimodalRoPEOp.hpp"

#include "mllm/utils/CPUArchHelper.hpp"
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
// Include AVX, SSE.
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <arm_neon.h>
#endif

namespace mllm::cpu {

Tensor Qwen2VLMultimodalRoPEOpImpl::makeInvFreq(int output_dim, float rope_theta) {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) { inv_freq_ptr[i] = 1.0 / std::pow(rope_theta, 2.0 * i / output_dim); }
  return inv_freq;
}

std::pair<Tensor, Tensor> Qwen2VLMultimodalRoPEOpImpl::makePositionEmbedding(Tensor& position_ids, Tensor& inv_freq,
                                                                             int seq_len, int output_dim,
                                                                             std::vector<int32_t>& mrope_section) {
  // Position ids shape is [3, 1, seq]
  MLLM_RT_ASSERT_EQ(position_ids.shape().size(), 3);
  MLLM_RT_ASSERT_EQ(position_ids.shape()[1], 1);  // Batch size is always 1.

  // [3, seq, dim]
  Tensor tmp_sin = Tensor::empty({3, position_ids.shape()[2], inv_freq.shape()[0] * 2}).alloc();
  Tensor tmp_cos = Tensor::empty({3, position_ids.shape()[2], inv_freq.shape()[0] * 2}).alloc();

  for (int b = 0; b < 3; ++b) {
    for (int d = 0; d < inv_freq.shape()[0]; ++d) {
      for (int s = 0; s < position_ids.shape()[2]; ++s) {
        auto value = inv_freq.ptr<float>()[d] * (*position_ids.offsettedPtr<int64_t>({b, 0, s}));
        *tmp_cos.offsettedPtr<float>({b, s, d}) = cosf(value);
        *tmp_cos.offsettedPtr<float>({b, s, d + inv_freq.shape()[0]}) = cosf(value);
        *tmp_sin.offsettedPtr<float>({b, s, d}) = sinf(value);
        *tmp_sin.offsettedPtr<float>({b, s, d + inv_freq.shape()[0]}) = sinf(value);
      }
    }
  }

  Tensor sin = Tensor::nil();
  Tensor cos = Tensor::nil();

  // mrope is always [16, 24, 24]
  if (!mrope_section.empty()) {
    auto double_rope_section = mrope_section;
    for (int i : mrope_section) { double_rope_section.push_back(i); }

    int num_rows = tmp_sin.shape()[1];
    int num_cols = tmp_sin.shape()[2];

    sin = Tensor::empty({num_rows, num_cols}, kFloat32, kCPU).alloc();
    cos = Tensor::empty({num_rows, num_cols}, kFloat32, kCPU).alloc();

    std::vector<int> start_cols;
    int current_start = 0;
    start_cols.push_back(current_start);
    for (int s : double_rope_section) {
      current_start += s;
      start_cols.push_back(current_start);
    }

    for (int j = 0; j < double_rope_section.size(); ++j) {
      int layer = j % 3;
      int s_j = double_rope_section[j];
      int start_col_in = start_cols[j];
      int start_col_out = start_cols[j];
      for (int row = 0; row < num_rows; ++row) {
        // Process cos
        auto in_cos_row_ptr = tmp_cos.offsettedPtr<float>({layer, row, 0});
        auto out_cos_row_ptr = cos.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) { out_cos_row_ptr[start_col_out + c] = in_cos_row_ptr[start_col_in + c]; }

        // Process sin
        auto in_sin_row_ptr = tmp_sin.offsettedPtr<float>({layer, row, 0});
        auto out_sin_row_ptr = sin.offsettedPtr<float>({row, 0});
        for (int c = 0; c < s_j; ++c) { out_sin_row_ptr[start_col_out + c] = in_sin_row_ptr[start_col_in + c]; }
      }
    }
  } else {
    sin = tmp_sin;
    cos = tmp_cos;
  }

  return {sin, cos};
}

void Qwen2VLMultimodalRoPEOpImpl::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Tensor& sin,
                                          Tensor& cos) {
  auto activation = inputs[0];
  auto out = outputs[0];

  // Activation must in BHSD layout
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  auto B = activation.shape()[0];
  auto H = activation.shape()[1];
  auto S = activation.shape()[2];
  auto D = activation.shape()[3];

  int32_t partial_dimension = D;
  int32_t half = D / 2;

  switch (activation.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("Qwen2VLVisionRoPEOpImpl is not implemented for x86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      for (int n = 0; n < B; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            float* act_ptr = activation.offsettedPtr<float>({n, h, s, 0});
            float* out_ptr = out.offsettedPtr<float>({n, h, s, 0});
            const float* sin_ptr = sin.offsettedPtr<float>({s, 0});
            const float* cos_ptr = cos.offsettedPtr<float>({s, 0});

            // Vectorized processing (4 elements per iteration)
            int d = 0;
            constexpr int step = 4;
            for (; d <= half - step; d += step) {
              // Load activation blocks
              float32x4_t act_front = vld1q_f32(act_ptr + d);
              float32x4_t act_back = vld1q_f32(act_ptr + d + half);

              // Load sin/cos values
              float32x4_t sin_vec = vld1q_f32(sin_ptr + d);
              float32x4_t cos_vec = vld1q_f32(cos_ptr + d);

              // Compute rotated values
              float32x4_t out_front = vsubq_f32(vmulq_f32(act_front, cos_vec), vmulq_f32(act_back, sin_vec));

              float32x4_t out_back = vaddq_f32(vmulq_f32(act_front, sin_vec), vmulq_f32(act_back, cos_vec));

              // Store results
              vst1q_f32(out_ptr + d, out_front);
              vst1q_f32(out_ptr + d + half, out_back);
            }

            // Process remaining elements
            for (; d < half; ++d) {
              float in_val = act_ptr[d];
              float in_val2 = act_ptr[d + half];
              float sin_val = sin_ptr[d];
              float cos_val = cos_ptr[d];

              out_ptr[d] = in_val * cos_val - in_val2 * sin_val;
              out_ptr[d + half] = in_val * sin_val + in_val2 * cos_val;
            }
          }
        }
      }
      break;
#endif
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      NYI("Qwen2VLVisionRoPEOpImpl is not implemented for x86");
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      for (int n = 0; n < B; ++n) {
        for (int h = 0; h < H; ++h) {
          for (int s = 0; s < S; ++s) {
            float16_t* act_ptr = activation.offsettedPtr<float16_t>({n, h, s, 0});
            float16_t* out_ptr = out.offsettedPtr<float16_t>({n, h, s, 0});
            const float16_t* sin_ptr = sin.offsettedPtr<float16_t>({s, 0});
            const float16_t* cos_ptr = cos.offsettedPtr<float16_t>({s, 0});

            // Vectorized processing (8 elements per iteration)
            int d = 0;
            constexpr int step = 8;
            for (; d <= half - step; d += step) {
              // Load activation blocks
              float16x8_t act_front = vld1q_f16(act_ptr + d);
              float16x8_t act_back = vld1q_f16(act_ptr + d + half);

              // Load sin/cos values
              float16x8_t sin_vec = vld1q_f16(sin_ptr + d);
              float16x8_t cos_vec = vld1q_f16(cos_ptr + d);

              // Compute rotated values
              float16x8_t out_front = vsubq_f16(vmulq_f16(act_front, cos_vec), vmulq_f16(act_back, sin_vec));

              float16x8_t out_back = vaddq_f16(vmulq_f16(act_front, sin_vec), vmulq_f16(act_back, cos_vec));

              // Store results
              vst1q_f16(out_ptr + d, out_front);
              vst1q_f16(out_ptr + d + half, out_back);
            }

            // Process remaining elements
            for (; d < half; ++d) {
              float in_val = static_cast<float>(act_ptr[d]);
              float in_val2 = static_cast<float>(act_ptr[d + half]);
              float sin_val = static_cast<float>(sin_ptr[d]);
              float cos_val = static_cast<float>(cos_ptr[d]);

              out_ptr[d] = static_cast<float16_t>(in_val * cos_val - in_val2 * sin_val);
              out_ptr[d + half] = static_cast<float16_t>(in_val * sin_val + in_val2 * cos_val);
            }
          }
        }
      }
#endif
      break;
    }
    default: {
      NYI("Qwen2VLMultimodalRoPEOpImpl::forward not support this dtype")
      break;
    }
  }
}

CPUMultimodalRoPEOp::CPUMultimodalRoPEOp(const aops::MultimodalRoPEOpOptions& options) : aops::MultimodalRoPEOp(options) {}

void CPUMultimodalRoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& activation = inputs[0];

  // Expect 2 inputs:
  // Pos 1: activations
  // Pos 2: position_ids
  MLLM_RT_ASSERT_EQ(inputs.size(), 2);

  // Input must be [B, H, S, D]
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  auto position_ids = inputs[1];
  auto out = outputs[0];

  switch (options_.type) {
    case aops::MultimodalRoPEOpOptionsType::kQwen2VL: {
      auto impl = Qwen2VLMultimodalRoPEOpImpl();

      Tensor sin = Tensor::nil();
      Tensor cos = Tensor::nil();

      if (inputs.size() > 2) {
        sin = inputs[2];
        cos = inputs[3];
      }

      if (!sin && !cos) {
        auto inv_freq = impl.makeInvFreq(activation.shape()[3], options_.qwen2vl_options.rope_theta);
        auto [_sin, _cos] = impl.makePositionEmbedding(position_ids, inv_freq, options_.qwen2vl_options.max_position_embeddings,
                                                       activation.shape()[3], options_.qwen2vl_options.mrope_section);
        sin = _sin;
        cos = _cos;

        sin = sin.to(activation.dtype());
        cos = cos.to(activation.dtype());
      }
      impl.forward(inputs, outputs, sin, cos);

      outputs.emplace_back(sin);
      outputs.emplace_back(cos);

      break;
    }
    default: {
      NYI("Unsupported");
      break;
    }
  }
}

}  // namespace mllm::cpu
