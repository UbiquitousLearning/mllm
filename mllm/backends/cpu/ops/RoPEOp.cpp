// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

namespace {

struct RoPENonFiniteSummary {
  size_t count = 0;
  int32_t first_b = -1;
  int32_t first_h = -1;
  int32_t first_s = -1;
  int32_t first_d = -1;
  uint32_t first_bits = 0;
};

bool ropeNonFiniteDiagnosticEnabled() {
  const char* value = std::getenv("MLLM_ROPE_NONFINITE_DIAGNOSTIC");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

RoPENonFiniteSummary scanRoPEF32(Tensor tensor, aops::RoPEOpOptionsInputType input_type) {
  RoPENonFiniteSummary summary;
  if (tensor.dtype() != kFloat32 || tensor.rank() != 4) { return summary; }

  const auto& shape = tensor.shape();
  const int32_t B = shape[0];
  const int32_t H = input_type == aops::RoPEOpOptionsInputType::kBHSD ? shape[1] : shape[2];
  const int32_t S = input_type == aops::RoPEOpOptionsInputType::kBHSD ? shape[2] : shape[1];
  const int32_t D = shape[3];
  for (int32_t b = 0; b < B; ++b) {
    for (int32_t h = 0; h < H; ++h) {
      for (int32_t s = 0; s < S; ++s) {
        const auto coordinates = input_type == aops::RoPEOpOptionsInputType::kBHSD
                                     ? std::vector<int>{b, h, s, 0}
                                     : std::vector<int>{b, s, h, 0};
        const auto* row = tensor.offsettedPtr<mllm_fp32_t>(coordinates);
        for (int32_t d = 0; d < D; ++d) {
          uint32_t bits = 0;
          std::memcpy(&bits, row + d, sizeof(bits));
          if ((bits & 0x7fffffffU) < 0x7f800000U) { continue; }
          if (summary.count == 0) {
            summary.first_b = b;
            summary.first_h = h;
            summary.first_s = s;
            summary.first_d = d;
            summary.first_bits = bits;
          }
          ++summary.count;
        }
      }
    }
  }
  return summary;
}

}  // namespace

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

  RoPENonFiniteSummary input_summary;
  const bool diagnose_nonfinite = ropeNonFiniteDiagnosticEnabled() && activation.dtype() == kFloat32;
  if (diagnose_nonfinite) { input_summary = scanRoPEF32(activation, options_.input_type); }

  auto impl = RoPEOpImpl();
  impl.forward(inputs, outputs, sin, cos, options_.partial_dim, options_.input_type);

  if (diagnose_nonfinite) {
    const auto output_summary = scanRoPEF32(out, options_.input_type);
    std::fprintf(stderr,
                 "MLLM_ROPE_NONFINITE name=%s input=%zu output=%zu "
                 "first_input=[%d,%d,%d,%d] input_bits=%u "
                 "first_output=[%d,%d,%d,%d] output_bits=%u\n",
                 getName().c_str(), input_summary.count, output_summary.count, input_summary.first_b, input_summary.first_h,
                 input_summary.first_s, input_summary.first_d, input_summary.first_bits, output_summary.first_b,
                 output_summary.first_h, output_summary.first_s, output_summary.first_d, output_summary.first_bits);
  }
}

}  // namespace mllm::cpu
