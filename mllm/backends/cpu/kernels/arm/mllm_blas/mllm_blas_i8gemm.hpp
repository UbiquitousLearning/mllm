// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

// NOTE from mllm team.
// Bitspack implement wxa32 quantization algorithms. It receive per-channel int8 activations and x bits per group quantized
// weight as inputs, output fp32 activations.

#pragma once

#include "mllm/utils/Common.hpp"
#include "mllm/backends/cpu/kernels/arm/quantize/bitspack/bitspack.hpp"
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_i8gemm_bitspack_1x1x32_fp32_neondot.hpp"
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_i8gemm_bitspack_1x4x16_fp32_neondot.hpp"
#include "mllm/backends/cpu/kernels/arm/mllm_blas/mllm_blas_i8gemm_bitspack_1x8x16_fp32_neondot.hpp"

namespace mllm::cpu::arm {  // NOLINT

namespace fp32_channel_a8_weight_group_ux {

inline size_t packed_activations_size(int m, int k, int group_size, bool has_weight_zeros, int mr, int kr, int sr) {
  (void)mr;  // unused
  (void)kr;  // unused
  (void)sr;  // unused
  return bitspack::activation_packing::packed_activations_size(m, k, group_size, has_weight_zeros);
}

inline size_t packed_activations_offset(int m_idx, int k, int group_size, bool has_weight_zeros, int mr, int kr, int sr) {
  assert(m_idx % mr == 0);
  auto packed_activations_size_mr_rows = packed_activations_size(mr, k, group_size, has_weight_zeros, mr, kr, sr);
  return (m_idx / mr) * packed_activations_size_mr_rows;
}

template<int mr_, int kr_, int sr_>
void pack_activations(void* packed_activations, int m, int k, int group_size, const float* activations, bool has_weight_zeros,
                      int mr, int kr, int sr) {
  (void)mr;  // unused
  (void)kr;  // unused
  (void)sr;  // unused
  bitspack::activation_packing::pack_activations<mr_, kr_, sr_>(packed_activations, m, k, group_size, activations,
                                                                has_weight_zeros);
}

inline size_t packed_weights_size(int n, int k, int group_size, int weight_nbit, bool has_weight_zeros, bool has_bias, int nr,
                                  int kr, int sr) {
  (void)kr;  // unused
  (void)sr;  // unused
  return bitspack::weight_packing::packed_weights_size(n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr);
}

inline size_t packed_weights_offset(int n_idx, int k, int group_size, int weight_nbit, bool has_weight_zeros, bool has_bias,
                                    int nr, int kr, int sr) {
  assert(n_idx % nr == 0);
  auto packed_weights_size_nr_cols =
      packed_weights_size(nr, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr);
  return (n_idx / nr) * packed_weights_size_nr_cols;
}

template<int weight_nbit, int nr_, int kr_, int sr_>
void pack_weights(void* packed_weights, int n, int k, int group_size, const int8_t* weight_qvals, const float* weight_scales,
                  const int8_t* weight_zeros, const float* bias, int nr, int kr, int sr) {
  (void)nr;  // unused
  (void)kr;  // unused
  (void)sr;  // unused
  bitspack::weight_packing::pack_weights<weight_nbit, nr_, kr_, sr_>(packed_weights, n, k, group_size, weight_qvals,
                                                                     weight_scales, weight_zeros, bias);
}

template<int weight_nbit, int nr_, int kr_, int sr_>
void pack_weights_with_lut(
    // Output
    void* packed_weights,
    // Inputs
    int n, int k, int group_size, const int8_t* weight_qval_idxs, int n_luts, const int8_t* luts, const float* weight_scales,
    // weight_zeros not packed if nullptr
    const int8_t* weight_zeros,
    // bias not packed if nullptr
    const float* bias, int nr, int kr, int sr) {
  (void)nr;  // unused
  (void)kr;  // unused
  (void)sr;  // unused
  bitspack::weight_packing::pack_weights_with_lut<weight_nbit, nr_, kr_, sr_>(
      packed_weights, n, k, group_size, weight_qval_idxs, n_luts, luts, weight_scales, weight_zeros, bias);
}

inline size_t packed_weights_with_lut_size(int n, int k, int group_size, int weight_nbit, bool has_weight_zeros, bool has_bias,
                                           int nr, int kr, int sr) {
  (void)kr;  // unused
  (void)sr;  // unused
  return bitspack::weight_packing::packed_weights_with_lut_size(n, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr);
}

inline size_t packed_weights_with_lut_offset(int n_idx, int k, int group_size, int weight_nbit, bool has_weight_zeros,
                                             bool has_bias, int nr, int kr, int sr) {
  assert(n_idx % nr == 0);
  auto packed_weights_size_nr_cols =
      packed_weights_with_lut_size(nr, k, group_size, weight_nbit, has_weight_zeros, has_bias, nr, kr, sr);
  return (n_idx / nr) * packed_weights_size_nr_cols;
}

template<int weight_nbit>
void kernel_1x1x32_f32_neondot(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride, int m, int n, int k, int group_size, const void* packed_weights, const void* packed_activations,
    // Ignored if has_clamp = false
    float clamp_min, float clamp_max, bool has_weight_zeros, bool has_bias, bool has_clamp) {
  kernel::kernel_1x1x32_f32_neondot<weight_nbit>(output, output_m_stride, m, n, k, group_size, packed_weights,
                                                 packed_activations, clamp_min, clamp_max, has_weight_zeros, has_bias,
                                                 has_clamp);
}

template<int weight_nbit>
void kernel_1x4x16_f32_neondot(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride, int m, int n, int k, int group_size, const void* packed_weights, const void* packed_activations,
    // Ignored if has_clamp = false
    float clamp_min, float clamp_max, bool has_weight_zeros, bool has_bias, bool has_clamp) {
  kernel::kernel_1x4x16_f32_neondot<weight_nbit>(output, output_m_stride, m, n, k, group_size, packed_weights,
                                                 packed_activations, clamp_min, clamp_max, has_weight_zeros, has_bias,
                                                 has_clamp);
}

template<int weight_nbit, bool has_weight_zeros, bool has_lut>
void kernel_1x8x16_f32_neondot(
    // Outputs
    float32_t* output,
    // Inputs
    int output_m_stride, int m, int n, int k, int group_size, const void* packed_weights, const void* packed_activations,
    // Ignored if has_clamp = false
    float clamp_min, float clamp_max, bool has_weight_zeros_, bool has_bias, bool has_clamp) {
  (void)has_weight_zeros_;  // unused
  kernel::kernel_1x8x16_f32_neondot<weight_nbit, has_weight_zeros, has_lut>(output, output_m_stride, m, n, k, group_size,
                                                                            packed_weights, packed_activations, clamp_min,
                                                                            clamp_max, has_bias, has_clamp);
}

}  // namespace fp32_channel_a8_weight_group_ux

struct MllmBlasI8UkernelConfig {
  // Size of packed_activations buffer
  using packed_activations_size_fn_type = size_t (*)(int m, int k, int group_size, bool has_weight_zeros, int mr, int kr,
                                                     int sr);

  // Offset in packed_activations buffer for a given m_idx
  // m_idx is index in unpacked activations matrix; it will be a multiple of
  // m_step
  using packed_activations_offset_fn_type = size_t (*)(int m_idx, int k, int group_size, bool has_weight_zeros, int mr, int kr,
                                                       int sr);

  // Pack activations into packed_activations buffer
  using pack_activations_fn_type = void (*)(void* packed_activations, int m, int k, int group_size, const float* activations,
                                            bool has_weight_zeros, int mr, int kr, int sr);

  // Size of packed_weights buffer
  using packed_weights_size_fn_type = size_t (*)(int n, int k, int group_size, int weight_nbit, bool has_weight_zeros,
                                                 bool has_bias, int nr, int kr, int sr);

  // Offset in packed_weights buffer for a given n_idx
  // n_inx is index in unpacked weights matrix; it will be a multiple of n_step
  using packed_weights_offset_fn_type = size_t (*)(int n_idx, int k, int group_size, int weight_nbit, bool has_weight_zeros,
                                                   bool has_bias, int nr, int kr, int sr);

  // Pack weights into packed_weights buffer
  using pack_weights_fn_type = void (*)(void* packed_weights, int n, int k, int group_size, const int8_t* weight_qvals,
                                        const float* weight_scales, const int8_t* weight_zeros, const float* bias, int nr,
                                        int kr, int sr);

  // Pack weights into packed_weights buffer with int8-valued LUT
  using pack_weights_with_lut_fn_type = void (*)(void* packed_weights, int n, int k, int group_size,
                                                 const int8_t* weight_qval_idxs, int n_luts, const int8_t* luts,
                                                 const float* weight_scales, const int8_t* weight_zeros, const float* bias,
                                                 int nr, int kr, int sr);

  // Run matmul kernel
  using kernel_fn_type = void (*)(float* output, int output_m_stride, int m, int n, int k, int group_size,
                                  const void* packed_weights, const void* packed_activations, float clamp_min, float clamp_max,
                                  bool has_weight_zeros, bool has_bias, bool has_clamp);

  struct linear_config_type {
    int m_step{0};  // m_idx will be a multiple of this
    int mr{0};
    packed_activations_size_fn_type packed_activations_size{nullptr};
    packed_activations_offset_fn_type packed_activations_offset{nullptr};
    pack_activations_fn_type pack_activations{nullptr};
    kernel_fn_type kernel{nullptr};
  };

  // preferred_alignment for packed_activations and packed_weights
  // Integration surfaces are not required to respect this alignment, and the
  // kernel must behave correctly no matter how buffers are aligned
  size_t preferred_alignment{0};
  int n_step{0};  // n_idx will be a multiple of this
  int nr{0};
  int kr{0};
  int sr{0};
  int weight_nbit{0};
  bool has_weight_zeros{false};
  bool has_bias{false};
  packed_weights_size_fn_type packed_weights_size{nullptr};
  packed_weights_offset_fn_type packed_weights_offset{nullptr};
  pack_weights_fn_type pack_weights{nullptr};
  pack_weights_with_lut_fn_type pack_weights_with_lut{nullptr};

  // linear_configs must be sorted in ascending m_step
  std::array<linear_config_type, 4> linear_configs;

  static MllmBlasI8UkernelConfig make(size_t preferred_alignment, int n_step, int nr, int kr, int sr, int weight_nbit,
                                      bool has_weight_zeros, bool has_bias, packed_weights_size_fn_type packed_weights_size,
                                      packed_weights_offset_fn_type packed_weights_offset, pack_weights_fn_type pack_weights,
                                      std::array<linear_config_type, 4> linear_configs);

  static MllmBlasI8UkernelConfig make_with_lut(size_t preferred_alignment, int n_step, int nr, int kr, int sr, int weight_nbit,
                                               bool has_weight_zeros, bool has_bias,
                                               packed_weights_size_fn_type packed_weights_with_lut_size,
                                               packed_weights_offset_fn_type packed_weights_with_lut_offset,
                                               pack_weights_with_lut_fn_type pack_weights_with_lut,
                                               std::array<linear_config_type, 4> linear_configs);

  inline void validate() const {
    MLLM_RT_ASSERT(preferred_alignment >= 1);
    MLLM_RT_ASSERT(n_step >= 1);
    MLLM_RT_ASSERT(nr >= 1);
    MLLM_RT_ASSERT(kr >= 1);
    MLLM_RT_ASSERT(sr >= 1);
    MLLM_RT_ASSERT(weight_nbit >= 1);
    MLLM_RT_ASSERT(packed_weights_size != nullptr);
    MLLM_RT_ASSERT(packed_weights_offset != nullptr);
    MLLM_RT_ASSERT(pack_weights != nullptr || pack_weights_with_lut != nullptr);

    bool linear_configs_set = true;  // first linear config must be set
    for (size_t i = 0; i < linear_configs.size(); i++) {
      if (linear_configs_set) {
        MLLM_RT_ASSERT(linear_configs[i].m_step >= 1);
        MLLM_RT_ASSERT(linear_configs[i].mr >= 1);
        MLLM_RT_ASSERT(linear_configs[i].packed_activations_size != nullptr);
        MLLM_RT_ASSERT(linear_configs[i].packed_activations_offset != nullptr);
        MLLM_RT_ASSERT(linear_configs[i].pack_activations != nullptr);
        MLLM_RT_ASSERT(linear_configs[i].kernel != nullptr);
        if (i >= 1) { MLLM_RT_ASSERT(linear_configs[i - 1].m_step < linear_configs[i].m_step); }
        if (i + 1 < linear_configs.size()) { linear_configs_set = (linear_configs[i + 1].m_step >= 1); }
      }
    }
  }

  [[nodiscard]] inline int select_linear_config_idx(int m) const {
    assert(m >= 1);
    assert(linear_configs[0].m_step >= 1);

    size_t i = 0;
    while (i + 1 < linear_configs.size() && linear_configs[i + 1].m_step >= 1 && linear_configs[i + 1].m_step <= m) {
      assert(linear_configs[i].m_step < linear_configs[i + 1].m_step);
      i++;
    }

    assert(i < linear_configs.size());
    assert(linear_configs[i].m_step >= 1);
    assert(i == 0 || linear_configs[i].m_step <= m);
    return static_cast<int>(i);
  }
};

inline MllmBlasI8UkernelConfig MllmBlasI8UkernelConfig::make(size_t preferred_alignment, int n_step, int nr, int kr, int sr,
                                                             int weight_nbit, bool has_weight_zeros, bool has_bias,
                                                             packed_weights_size_fn_type packed_weights_size,
                                                             packed_weights_offset_fn_type packed_weights_offset,
                                                             pack_weights_fn_type pack_weights,
                                                             std::array<linear_config_type, 4> linear_configs) {
  return MllmBlasI8UkernelConfig{.preferred_alignment = preferred_alignment,
                                 .n_step = n_step,
                                 .nr = nr,
                                 .kr = kr,
                                 .sr = sr,
                                 .weight_nbit = weight_nbit,
                                 .has_weight_zeros = has_weight_zeros,
                                 .has_bias = has_bias,
                                 .packed_weights_size = packed_weights_size,
                                 .packed_weights_offset = packed_weights_offset,
                                 .pack_weights = pack_weights,
                                 /*pack_weights_with_lut*/ .pack_weights_with_lut = nullptr,
                                 .linear_configs = linear_configs};
}

inline MllmBlasI8UkernelConfig MllmBlasI8UkernelConfig::make_with_lut(
    size_t preferred_alignment, int n_step, int nr, int kr, int sr, int weight_nbit, bool has_weight_zeros, bool has_bias,
    packed_weights_size_fn_type packed_weights_with_lut_size, packed_weights_offset_fn_type packed_weights_with_lut_offset,
    pack_weights_with_lut_fn_type pack_weights_with_lut, std::array<linear_config_type, 4> linear_configs) {
  return MllmBlasI8UkernelConfig{.preferred_alignment = preferred_alignment,
                                 .n_step = n_step,
                                 .nr = nr,
                                 .kr = kr,
                                 .sr = sr,
                                 .weight_nbit = weight_nbit,
                                 .has_weight_zeros = has_weight_zeros,
                                 .has_bias = has_bias,
                                 .packed_weights_size = packed_weights_with_lut_size,
                                 .packed_weights_offset = packed_weights_with_lut_offset,
                                 /*pack_weights*/ .pack_weights = nullptr,
                                 /*pack_weights_with_lut*/ .pack_weights_with_lut = pack_weights_with_lut,
                                 .linear_configs = linear_configs};
}

}  // namespace mllm::cpu::arm
