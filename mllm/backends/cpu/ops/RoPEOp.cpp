// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstring>
#include "mllm/backends/cpu/ops/RoPEOp.hpp"

#include "mllm/core/aops/RoPEOp.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
// Include AVX, SSE.
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include <arm_neon.h>
#endif

namespace mllm::cpu {

void RoPEOpImpl::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, Tensor& sin, Tensor& cos,
                         int32_t partial_dim, aops::RoPEOpOptionsInputType input_layout_type) {
  auto activation = inputs[0];
  auto out = outputs[0];

  // Activation must in BHSD or BSHD layout
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);

  auto B = 0;
  auto H = 0;
  auto S = 0;
  auto D = 0;

  switch (input_layout_type) {
    case aops::RoPEOpOptionsInputType::kBHSD: {
      B = activation.shape()[0];
      H = activation.shape()[1];
      S = activation.shape()[2];
      D = activation.shape()[3];
      break;
    }
    case aops::RoPEOpOptionsInputType::kBSHD: {
      B = activation.shape()[0];
      S = activation.shape()[1];
      H = activation.shape()[2];
      D = activation.shape()[3];
      break;
    }
  }

  int32_t half = D / 2;
  if (partial_dim != -1) { half = partial_dim / 2; }
  int32_t tail_D_start = half * 2;
  int32_t tail_D_end = D;

  switch (activation.dtype()) {
    case kFloat32: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      switch (input_layout_type) {
        case aops::RoPEOpOptionsInputType::kBHSD: {
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                mllm_fp32_t* act_ptr = activation.offsettedPtr<mllm_fp32_t>({n, h, s, 0});
                mllm_fp32_t* out_ptr = out.offsettedPtr<mllm_fp32_t>({n, h, s, 0});
                const mllm_fp32_t* sin_ptr = sin.offsettedPtr<mllm_fp32_t>({n, s, 0});
                const mllm_fp32_t* cos_ptr = cos.offsettedPtr<mllm_fp32_t>({n, s, 0});

                for (int d = 0; d < half; ++d) {
                  mllm_fp32_t in_val = act_ptr[d];
                  mllm_fp32_t in_val2 = act_ptr[d + half];
                  mllm_fp32_t sin_val = sin_ptr[d];
                  mllm_fp32_t cos_val = cos_ptr[d];

                  out_ptr[d] = in_val * cos_val - in_val2 * sin_val;
                  out_ptr[d + half] = in_val * sin_val + in_val2 * cos_val;
                }
              }
            }
          }

          // Concat
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                mllm_fp32_t* act_ptr = activation.offsettedPtr<mllm_fp32_t>({n, h, s, 0});
                mllm_fp32_t* out_ptr = out.offsettedPtr<mllm_fp32_t>({n, h, s, 0});
                for (int d = tail_D_start; d < tail_D_end; ++d) {
                  mllm_fp32_t in_val = act_ptr[d];
                  out_ptr[d] = in_val;
                }
              }
            }
          }
          break;
        }
        case aops::RoPEOpOptionsInputType::kBSHD: {
          for (int n = 0; n < B; ++n) {
            for (int s = 0; s < S; ++s) {
              const mllm_fp32_t* sin_ptr = sin.offsettedPtr<mllm_fp32_t>({n, s, 0});
              const mllm_fp32_t* cos_ptr = cos.offsettedPtr<mllm_fp32_t>({n, s, 0});
              for (int h = 0; h < H; ++h) {
                mllm_fp32_t* act_ptr = activation.offsettedPtr<mllm_fp32_t>({n, s, h, 0});
                mllm_fp32_t* out_ptr = out.offsettedPtr<mllm_fp32_t>({n, s, h, 0});
                for (int d = 0; d < half; ++d) {
                  mllm_fp32_t in_val = act_ptr[d];
                  mllm_fp32_t in_val2 = act_ptr[d + half];
                  mllm_fp32_t sin_val = sin_ptr[d];
                  mllm_fp32_t cos_val = cos_ptr[d];
                  out_ptr[d] = in_val * cos_val - in_val2 * sin_val;
                  out_ptr[d + half] = in_val * sin_val + in_val2 * cos_val;
                }
              }
            }
          }

          // concat
          for (int n = 0; n < B; ++n) {
            for (int s = 0; s < S; ++s) {
              for (int h = 0; h < H; ++h) {
                mllm_fp32_t* act_ptr = activation.offsettedPtr<mllm_fp32_t>({n, s, h, 0});
                mllm_fp32_t* out_ptr = out.offsettedPtr<mllm_fp32_t>({n, s, h, 0});
                for (int d = tail_D_start; d < tail_D_end; ++d) {
                  mllm_fp32_t in_val = act_ptr[d];
                  out_ptr[d] = in_val;
                }
              }
            }
          }
          break;
        }
      }
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      switch (input_layout_type) {
        case aops::RoPEOpOptionsInputType::kBHSD: {
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                mllm_fp32_t* act_ptr = activation.offsettedPtr<mllm_fp32_t>({n, h, s, 0});
                mllm_fp32_t* out_ptr = out.offsettedPtr<mllm_fp32_t>({n, h, s, 0});
                const mllm_fp32_t* sin_ptr = sin.offsettedPtr<mllm_fp32_t>({n, s, 0});
                const mllm_fp32_t* cos_ptr = cos.offsettedPtr<mllm_fp32_t>({n, s, 0});

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
                  mllm_fp32_t in_val = act_ptr[d];
                  mllm_fp32_t in_val2 = act_ptr[d + half];
                  mllm_fp32_t sin_val = sin_ptr[d];
                  mllm_fp32_t cos_val = cos_ptr[d];

                  out_ptr[d] = in_val * cos_val - in_val2 * sin_val;
                  out_ptr[d + half] = in_val * sin_val + in_val2 * cos_val;
                }
              }
            }
          }

          // may need concat
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                mllm_fp32_t* act_ptr = activation.offsettedPtr<mllm_fp32_t>({n, h, s, 0});
                mllm_fp32_t* out_ptr = out.offsettedPtr<mllm_fp32_t>({n, h, s, 0});

                // Vectorized processing (4 elements per iteration)
                int d = tail_D_start;
                constexpr int step = 4;
                for (; d <= tail_D_end - step; d += step) {
                  // Load activation blocks
                  float32x4_t act_front = vld1q_f32(act_ptr + d);
                  vst1q_f32(out_ptr + d, act_front);
                }

                // Process remaining elements
                for (; d < tail_D_end; ++d) {
                  mllm_fp32_t in_val = act_ptr[d];
                  out_ptr[d] = in_val;
                }
              }
            }
          }
          break;
        }
        case aops::RoPEOpOptionsInputType::kBSHD: {
          constexpr int step = 4;
          for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
              const mllm_fp32_t* sin_ptr = sin.offsettedPtr<mllm_fp32_t>({b, s, 0});
              const mllm_fp32_t* cos_ptr = cos.offsettedPtr<mllm_fp32_t>({b, s, 0});

              for (int h = 0; h < H; ++h) {
                mllm_fp32_t* act_ptr = activation.offsettedPtr<mllm_fp32_t>({b, s, h, 0});
                mllm_fp32_t* out_ptr = out.offsettedPtr<mllm_fp32_t>({b, s, h, 0});

                int d = 0;
                for (; d <= half - step; d += step) {
                  float32x4_t act_front = vld1q_f32(act_ptr + d);        // [d, d+1, d+2, d+3]
                  float32x4_t act_back = vld1q_f32(act_ptr + d + half);  // [d+half, ...]
                  float32x4_t sin_vec = vld1q_f32(sin_ptr + d);
                  float32x4_t cos_vec = vld1q_f32(cos_ptr + d);

                  float32x4_t out_front = vsubq_f32(vmulq_f32(act_front, cos_vec), vmulq_f32(act_back, sin_vec));
                  float32x4_t out_back = vaddq_f32(vmulq_f32(act_front, sin_vec), vmulq_f32(act_back, cos_vec));

                  vst1q_f32(out_ptr + d, out_front);
                  vst1q_f32(out_ptr + d + half, out_back);
                }
                for (; d < half; ++d) {
                  mllm_fp32_t in0 = act_ptr[d];
                  mllm_fp32_t in1 = act_ptr[d + half];
                  mllm_fp32_t s = sin_ptr[d];
                  mllm_fp32_t c = cos_ptr[d];

                  out_ptr[d] = in0 * c - in1 * s;
                  out_ptr[d + half] = in0 * s + in1 * c;
                }
              }
            }
          }

          // concat
          for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S; ++s) {
              for (int h = 0; h < H; ++h) {
                mllm_fp32_t* act_ptr = activation.offsettedPtr<mllm_fp32_t>({b, s, h, 0});
                mllm_fp32_t* out_ptr = out.offsettedPtr<mllm_fp32_t>({b, s, h, 0});

                int d = tail_D_start;
                for (; d <= tail_D_end - step; d += step) {
                  float32x4_t act_front = vld1q_f32(act_ptr + d);  // [d, d+1, d+2, d+3]
                  vst1q_f32(out_ptr + d, act_front);
                }
                for (; d < tail_D_end; ++d) {
                  mllm_fp32_t in0 = act_ptr[d];
                  out_ptr[d] = in0;
                }
              }
            }
          }
          break;
        }
      }
#endif
      break;
    }
    case kFloat16: {
#if defined(MLLM_HOST_ARCH_X86_64) || defined(MLLM_HOST_ARCH_X86)
      switch (input_layout_type) {
        case aops::RoPEOpOptionsInputType::kBHSD: {
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                mllm_fp16_t* act_ptr = activation.offsettedPtr<mllm_fp16_t>({n, h, s, 0});
                mllm_fp16_t* out_ptr = out.offsettedPtr<mllm_fp16_t>({n, h, s, 0});
                const mllm_fp16_t* sin_ptr = sin.offsettedPtr<mllm_fp16_t>({n, s, 0});
                const mllm_fp16_t* cos_ptr = cos.offsettedPtr<mllm_fp16_t>({n, s, 0});

                for (int d = 0; d < half; ++d) {
                  mllm_fp32_t in_val = static_cast<mllm_fp32_t>(act_ptr[d]);
                  mllm_fp32_t in_val2 = static_cast<mllm_fp32_t>(act_ptr[d + half]);
                  mllm_fp32_t sin_val = static_cast<mllm_fp32_t>(sin_ptr[d]);
                  mllm_fp32_t cos_val = static_cast<mllm_fp32_t>(cos_ptr[d]);

                  out_ptr[d] = static_cast<mllm_fp16_t>(in_val * cos_val - in_val2 * sin_val);
                  out_ptr[d + half] = static_cast<mllm_fp16_t>(in_val * sin_val + in_val2 * cos_val);
                }
              }
            }
          }

          // concat
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                mllm_fp16_t* act_ptr = activation.offsettedPtr<mllm_fp16_t>({n, h, s, 0});
                mllm_fp16_t* out_ptr = out.offsettedPtr<mllm_fp16_t>({n, h, s, 0});
                for (int d = tail_D_start; d < tail_D_end; ++d) {
                  mllm_fp32_t in_val = static_cast<mllm_fp32_t>(act_ptr[d]);
                  out_ptr[d] = static_cast<mllm_fp16_t>(in_val);
                }
              }
            }
          }
          break;
        }
        case aops::RoPEOpOptionsInputType::kBSHD: {
          for (int n = 0; n < B; ++n) {
            for (int s = 0; s < S; ++s) {
              const mllm_fp16_t* sin_ptr = sin.offsettedPtr<mllm_fp16_t>({n, s, 0});
              const mllm_fp16_t* cos_ptr = cos.offsettedPtr<mllm_fp16_t>({n, s, 0});
              for (int h = 0; h < H; ++h) {
                mllm_fp16_t* act_ptr = activation.offsettedPtr<mllm_fp16_t>({n, s, h, 0});
                mllm_fp16_t* out_ptr = out.offsettedPtr<mllm_fp16_t>({n, s, h, 0});
                for (int d = 0; d < half; ++d) {
                  mllm_fp32_t in_val = static_cast<mllm_fp32_t>(act_ptr[d]);
                  mllm_fp32_t in_val2 = static_cast<mllm_fp32_t>(act_ptr[d + half]);
                  mllm_fp32_t sin_val = static_cast<mllm_fp32_t>(sin_ptr[d]);
                  mllm_fp32_t cos_val = static_cast<mllm_fp32_t>(cos_ptr[d]);

                  out_ptr[d] = static_cast<mllm_fp16_t>(in_val * cos_val - in_val2 * sin_val);
                  out_ptr[d + half] = static_cast<mllm_fp16_t>(in_val * sin_val + in_val2 * cos_val);
                }
              }
            }
          }

          // concat
          for (int n = 0; n < B; ++n) {
            for (int s = 0; s < S; ++s) {
              for (int h = 0; h < H; ++h) {
                mllm_fp16_t* act_ptr = activation.offsettedPtr<mllm_fp16_t>({n, s, h, 0});
                mllm_fp16_t* out_ptr = out.offsettedPtr<mllm_fp16_t>({n, s, h, 0});
                for (int d = tail_D_start; d < tail_D_end; ++d) {
                  mllm_fp32_t in_val = static_cast<mllm_fp32_t>(act_ptr[d]);
                  out_ptr[d] = static_cast<mllm_fp16_t>(in_val);
                }
              }
            }
          }
          break;
        }
      }
#elif defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
      switch (input_layout_type) {
        case aops::RoPEOpOptionsInputType::kBHSD: {
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                mllm_fp16_t* act_ptr = activation.offsettedPtr<mllm_fp16_t>({n, h, s, 0});
                mllm_fp16_t* out_ptr = out.offsettedPtr<mllm_fp16_t>({n, h, s, 0});
                const mllm_fp16_t* sin_ptr = sin.offsettedPtr<mllm_fp16_t>({n, s, 0});
                const mllm_fp16_t* cos_ptr = cos.offsettedPtr<mllm_fp16_t>({n, s, 0});

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
                  mllm_fp32_t in_val = static_cast<mllm_fp32_t>(act_ptr[d]);
                  mllm_fp32_t in_val2 = static_cast<mllm_fp32_t>(act_ptr[d + half]);
                  mllm_fp32_t sin_val = static_cast<mllm_fp32_t>(sin_ptr[d]);
                  mllm_fp32_t cos_val = static_cast<mllm_fp32_t>(cos_ptr[d]);

                  out_ptr[d] = static_cast<mllm_fp16_t>(in_val * cos_val - in_val2 * sin_val);
                  out_ptr[d + half] = static_cast<mllm_fp16_t>(in_val * sin_val + in_val2 * cos_val);
                }
              }
            }
          }

          // concat
          for (int n = 0; n < B; ++n) {
            for (int h = 0; h < H; ++h) {
              for (int s = 0; s < S; ++s) {
                mllm_fp16_t* act_ptr = activation.offsettedPtr<mllm_fp16_t>({n, h, s, 0});
                mllm_fp16_t* out_ptr = out.offsettedPtr<mllm_fp16_t>({n, h, s, 0});

                // Vectorized processing (8 elements per iteration)
                int d = tail_D_start;
                constexpr int step = 8;
                for (; d <= tail_D_end - step; d += step) {
                  // Load activation blocks
                  float16x8_t act_front = vld1q_f16(act_ptr + d);
                  // Store results
                  vst1q_f16(out_ptr + d, act_front);
                }

                // Process remaining elements
                for (; d < tail_D_end; ++d) {
                  mllm_fp32_t in_val = static_cast<mllm_fp32_t>(act_ptr[d]);
                  out_ptr[d] = static_cast<mllm_fp16_t>(in_val);
                }
              }
            }
          }
          break;
        }
        case aops::RoPEOpOptionsInputType::kBSHD: {
          for (int n = 0; n < B; ++n) {
            for (int s = 0; s < S; ++s) {
              const mllm_fp16_t* sin_ptr = sin.offsettedPtr<mllm_fp16_t>({n, s, 0});
              const mllm_fp16_t* cos_ptr = cos.offsettedPtr<mllm_fp16_t>({n, s, 0});

              for (int h = 0; h < H; ++h) {
                mllm_fp16_t* act_ptr = activation.offsettedPtr<mllm_fp16_t>({n, s, h, 0});
                mllm_fp16_t* out_ptr = out.offsettedPtr<mllm_fp16_t>({n, s, h, 0});
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
                  mllm_fp32_t in_val = static_cast<mllm_fp32_t>(act_ptr[d]);
                  mllm_fp32_t in_val2 = static_cast<mllm_fp32_t>(act_ptr[d + half]);
                  mllm_fp32_t sin_val = static_cast<mllm_fp32_t>(sin_ptr[d]);
                  mllm_fp32_t cos_val = static_cast<mllm_fp32_t>(cos_ptr[d]);

                  out_ptr[d] = static_cast<mllm_fp16_t>(in_val * cos_val - in_val2 * sin_val);
                  out_ptr[d + half] = static_cast<mllm_fp16_t>(in_val * sin_val + in_val2 * cos_val);
                }
              }
            }
          }

          // concat
          for (int n = 0; n < B; ++n) {
            for (int s = 0; s < S; ++s) {
              for (int h = 0; h < H; ++h) {
                mllm_fp16_t* act_ptr = activation.offsettedPtr<mllm_fp16_t>({n, s, h, 0});
                mllm_fp16_t* out_ptr = out.offsettedPtr<mllm_fp16_t>({n, s, h, 0});
                // Vectorized processing (8 elements per iteration)
                int d = tail_D_start;
                constexpr int step = 8;
                for (; d <= tail_D_end - step; d += step) {
                  // Load activation blocks
                  float16x8_t act_front = vld1q_f16(act_ptr + d);
                  // Store results
                  vst1q_f16(out_ptr + d, act_front);
                }

                // Process remaining elements
                for (; d < tail_D_end; ++d) {
                  mllm_fp32_t in_val = static_cast<mllm_fp32_t>(act_ptr[d]);
                  out_ptr[d] = static_cast<mllm_fp16_t>(in_val);
                }
              }
            }
          }
          break;
        }
      }
#endif
      break;
    }
    default: {
      NYI("RoPEOpImpl::forward not support this dtype")
      break;
    }
  }
}

CPURoPEOp::CPURoPEOp(const aops::RoPEOpOptions& options) : aops::RoPEOp(options) {}

void CPURoPEOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Expect 3 inputs:
  // Pos 0: activations
  // Pos 1: sin
  // Pos 2: cos
  MLLM_RT_ASSERT_EQ(inputs.size(), 3);

  auto& activation = inputs[0];
  auto sin = inputs[1];
  auto cos = inputs[2];

  // Input must be [B, H, S, D]
  MLLM_RT_ASSERT_EQ(activation.shape().size(), 4);
  auto out = outputs[0];

  auto impl = RoPEOpImpl();
  impl.forward(inputs, outputs, sin, cos, options_.partial_dim, options_.input_type);
}

}  // namespace mllm::cpu
