// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/LinearShadowCorrectionOp.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "mllm/backends/qnn/custom-op-package/LLaMAPackage/src/ops/ShadowCorrectionPacketLayout.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::cpu {

namespace {

namespace packet = ::shadow_correction_packet;

constexpr int32_t kQuantMin = -32768;
constexpr int32_t kQuantMax = 32767;

bool needsOutputRepair(int16_t code) { return code == kQuantMin || code == kQuantMax; }

bool scalarFlag(const Tensor& tensor) {
  switch (tensor.dtype()) {
    case kBool:
    case kUInt8: return tensor.ptr<uint8_t>()[0] != 0;
    case kInt8: return tensor.ptr<int8_t>()[0] != 0;
    case kInt16: return tensor.ptr<int16_t>()[0] != 0;
    case kInt32: return tensor.ptr<int32_t>()[0] != 0;
    default: NYI("Unsupported clip flag dtype");
  }
  return false;
}

}  // namespace

CPULinearShadowCorrectionOp::CPULinearShadowCorrectionOp(const aops::LinearShadowCorrectionOpOptions& options)
    : LinearShadowCorrectionOp(options) {}

void CPULinearShadowCorrectionOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  // Keep the original Float32 loop separate from the packet consumer. The
  // different loop shapes otherwise change arithmetic through optimization.
  if (inputs[0].dtype() == kInt32) {
    forwardPacket(inputs, outputs);
    return;
  }
  forwardOriginal(inputs, outputs);
}

[[gnu::noinline]] void CPULinearShadowCorrectionOp::forwardOriginal(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  const auto& original_input = inputs[0];
  const auto& quantized_output = inputs[1];
  const auto& baseline = inputs[2];
  auto& corrected = outputs[0];

  std::memcpy(corrected.ptr<void>(), baseline.ptr<void>(), baseline.bytes());

  const bool clip_input = scalarFlag(clip_input_);
  const bool clip_output = scalarFlag(clip_output_);
  if (clip_input && clip_output) { return; }

  constexpr int32_t kQuantMin = -32768;
  constexpr int32_t kQuantMax = 32767;
  const float weight_step = weight_scale_.ptr<float>()[0];
  const float input_step = input_scale_.ptr<float>()[0] / static_cast<float>(kQuantMax);
  const float output_step = output_scale_.ptr<float>()[0] / static_cast<float>(kQuantMax);
  MLLM_RT_ASSERT(std::isfinite(weight_step) && weight_step > 0.0F);
  MLLM_RT_ASSERT(std::isfinite(input_step) && input_step > 0.0F);
  MLLM_RT_ASSERT(std::isfinite(output_step) && output_step > 0.0F);

  const auto* x = original_input.ptr<float>();
  const auto* q_output = quantized_output.ptr<int16_t>();
  const auto* base = baseline.ptr<float>();
  auto* output = corrected.ptr<float>();
  const auto* weight = weight_.ptr<int8_t>();
  const int32_t in_channels = options_.in_channels;
  const int32_t out_channels = options_.out_channels;
  const int64_t tokens = baseline.numel() / out_channels;

  for (int64_t token = 0; token < tokens; ++token) {
    const auto* token_input = x + token * in_channels;
    auto* token_output = output + token * out_channels;

    if (!clip_input) {
      for (int32_t input_channel = 0; input_channel < in_channels; ++input_channel) {
        const float unbounded_q = std::round(token_input[input_channel] / input_step);
        if (unbounded_q >= static_cast<float>(kQuantMin) && unbounded_q <= static_cast<float>(kQuantMax)) { continue; }
        const float clipped_q = std::clamp(unbounded_q, static_cast<float>(kQuantMin), static_cast<float>(kQuantMax));
        const float correction_scale = (unbounded_q - clipped_q) * input_step * weight_step;
        const auto* weight_row = weight + static_cast<int64_t>(input_channel) * out_channels;
        for (int32_t output_channel = 0; output_channel < out_channels; ++output_channel) {
          token_output[output_channel] += correction_scale * static_cast<float>(weight_row[output_channel]);
        }
      }
    }

    if (!clip_output) {
      const auto* token_q_output = q_output + token * out_channels;
      const auto* token_baseline = base + token * out_channels;
      for (int32_t output_channel = 0; output_channel < out_channels; ++output_channel) {
        const int32_t q_value = token_q_output[output_channel];
        if (q_value > kQuantMin && q_value < kQuantMax) { continue; }

        float full_sum = 0.0F;
        for (int32_t input_channel = 0; input_channel < in_channels; ++input_channel) {
          const float unbounded_q = std::round(token_input[input_channel] / input_step);
          const int8_t q_weight = weight[static_cast<int64_t>(input_channel) * out_channels + output_channel];
          full_sum += unbounded_q * input_step * static_cast<float>(q_weight) * weight_step;
        }
        const float restored_q = std::round(full_sum / output_step);
        token_output[output_channel] =
            token_baseline[output_channel] - static_cast<float>(q_value) * output_step + restored_q * output_step;
      }
    }
  }
}

void CPULinearShadowCorrectionOp::forwardPacket(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
#if defined(__clang__)
#pragma clang fp reassociate(off)
#endif
  const auto& original_input = inputs[0];
  const auto& quantized_output = inputs[1];
  const auto& baseline = inputs[2];
  auto& corrected = outputs[0];

  std::memcpy(corrected.ptr<void>(), baseline.ptr<void>(), baseline.bytes());

  const bool clip_input = scalarFlag(clip_input_);
  const bool clip_output = scalarFlag(clip_output_);
  const float weight_step = weight_scale_.ptr<float>()[0];
  const float input_scale = input_scale_.ptr<float>()[0];
  const float output_scale = output_scale_.ptr<float>()[0];
  const float input_step = input_scale * (1.0f / static_cast<float>(kQuantMax));
  const float output_step = output_scale * (1.0f / static_cast<float>(kQuantMax));
  const float inverse_input_scale = 1.0f / input_scale;
  const float dense_input_multiplier = static_cast<float>(kQuantMax) / input_scale;
  const float dense_output_multiplier = static_cast<float>(kQuantMax) / output_scale;
  const float weight_input_step = input_step * weight_step;
  MLLM_RT_ASSERT(std::isfinite(weight_step) && weight_step > 0.0F);
  MLLM_RT_ASSERT(std::isfinite(input_step) && input_step > 0.0F);
  MLLM_RT_ASSERT(std::isfinite(output_step) && output_step > 0.0F);

  const auto* q_output = quantized_output.ptr<int16_t>();
  const auto* base = baseline.ptr<float>();
  auto* output = corrected.ptr<float>();
  const auto* weight = weight_.ptr<int8_t>();
  const int32_t in_channels = options_.in_channels;
  const int32_t out_channels = options_.out_channels;
  const int64_t tokens = baseline.numel() / out_channels;

  const auto repair_input_channel = [&](int64_t token, uint32_t channel, float activation) {
    const float unbounded_q = std::round((activation * static_cast<float>(kQuantMax)) * inverse_input_scale);
    if (unbounded_q >= static_cast<float>(kQuantMin) && unbounded_q <= static_cast<float>(kQuantMax)) { return; }
    const float clipped_q = std::clamp(unbounded_q, static_cast<float>(kQuantMin), static_cast<float>(kQuantMax));
    const float correction_scale = (unbounded_q - clipped_q) * weight_input_step;
    const auto* weight_row = weight + static_cast<int64_t>(channel) * out_channels;
    auto* token_output = output + token * out_channels;
    for (int32_t output_channel = 0; output_channel < out_channels; ++output_channel) {
      token_output[output_channel] =
          std::fma(correction_scale, static_cast<float>(weight_row[output_channel]), token_output[output_channel]);
    }
  };

  const auto repair_dense_row = [&](int64_t token, const float* token_input) {
    if (!clip_input) {
      for (int32_t channel = 0; channel < in_channels; ++channel) {
        repair_input_channel(token, channel, token_input[channel]);
      }
    }
    if (clip_output) { return; }
    const auto* token_q_output = q_output + token * out_channels;
    const auto* token_baseline = base + token * out_channels;
    auto* token_output = output + token * out_channels;
    for (int32_t output_channel = 0; output_channel < out_channels; ++output_channel) {
      const int16_t q_value = token_q_output[output_channel];
      if (!needsOutputRepair(q_value)) { continue; }

      // The retained Android loop accumulates the even and odd channels
      // independently before restoring the output code.
      float even = 0.0f;
      float odd = 0.0f;
      for (int32_t channel = 0; channel < in_channels; channel += 2) {
        const auto* w = weight + static_cast<int64_t>(channel) * out_channels + output_channel;
        const float even_q = std::round(token_input[channel] * dense_input_multiplier);
        const float odd_q = std::round(token_input[channel + 1] * dense_input_multiplier);
        even = std::fma(even_q * weight_input_step, static_cast<float>(w[0]), even);
        odd = std::fma(odd_q * weight_input_step, static_cast<float>(w[out_channels]), odd);
      }
      const float restored_q = std::round((even + odd) * dense_output_multiplier);
      token_output[output_channel] =
          std::fma(restored_q - static_cast<float>(q_value), output_step, token_baseline[output_channel]);
    }
  };

  // Read serialized words instead of treating tensor storage as a live C++ struct.
  using Wire = packet::correction_packet;
  constexpr size_t word = sizeof(uint32_t);
  constexpr size_t offset_status = offsetof(Wire, status) / word;
  constexpr size_t offset_sparse_count = offsetof(Wire, sparse_count) / word;
  constexpr size_t offset_dense_count = offsetof(Wire, dense_count) / word;
  constexpr size_t offset_flags = offsetof(Wire, mask_flags) / word;
  constexpr size_t offset_rows = offsetof(Wire, row_offsets) / word;
  constexpr size_t offset_dense_ids = offsetof(Wire, dense_row_ids) / word;
  constexpr size_t offset_sparse = offsetof(Wire, sparse) / word;
  constexpr size_t offset_dense = offsetof(Wire, dense_rows) / word;
  constexpr size_t entry_words = sizeof(packet::sparse_entry) / word;
  constexpr size_t entry_channel = offsetof(packet::sparse_entry, channel) / word;
  constexpr size_t entry_value = offsetof(packet::sparse_entry, value_bits) / word;
  const uint32_t flags = (clip_input ? packet::clip_input : 0) | (clip_output ? packet::clip_output : 0);
  const auto* words = reinterpret_cast<const uint32_t*>(original_input.ptr<int32_t>());
  std::array<float, packet::in_width> activations;
  for (int64_t tile_start = 0; tile_start < tokens; tile_start += packet::tile_rows, words += packet::packet_words) {
    MLLM_RT_ASSERT_EQ(words[offset_status], packet::complete);
    MLLM_RT_ASSERT_EQ(words[offset_flags], flags);
    const uint32_t sparse_count = words[offset_sparse_count];
    const uint32_t dense_count = words[offset_dense_count];
    MLLM_RT_ASSERT(sparse_count <= packet::sparse_capacity && (!clip_input || sparse_count == 0));
    MLLM_RT_ASSERT(dense_count <= packet::dense_capacity && (!clip_output || dense_count == 0));
    const auto* rows = words + offset_rows;
    const auto* dense_ids = words + offset_dense_ids;
    const auto* sparse = words + offset_sparse;
    MLLM_RT_ASSERT(rows[0] == 0 && rows[packet::tile_rows] == sparse_count);

    uint32_t dense_index = 0;
    for (uint32_t row = 0; row < packet::tile_rows; ++row) {
      const int64_t token = tile_start + row;
      const uint32_t begin = rows[row];
      const uint32_t end = rows[row + 1];
      MLLM_RT_ASSERT(begin <= end && end <= sparse_count);
      const bool dense = dense_index < dense_count && dense_ids[dense_index] == row;
      if (!clip_output) {
        const auto* codes = q_output + token * out_channels;
        MLLM_RT_ASSERT(std::any_of(codes, codes + out_channels, needsOutputRepair) == dense);
      }
      if (dense) {
        MLLM_RT_ASSERT(begin == end);
        std::memcpy(activations.data(), words + offset_dense + dense_index * packet::in_width, sizeof(activations));
        repair_dense_row(token, activations.data());
        ++dense_index;
        continue;
      }
      int32_t last_channel = -1;
      for (uint32_t i = begin; i < end; ++i) {
        const auto* entry = sparse + i * entry_words;
        const uint32_t channel = entry[entry_channel];
        const float value = std::bit_cast<float>(entry[entry_value]);
        MLLM_RT_ASSERT(channel < packet::in_width && static_cast<int32_t>(channel) > last_channel && std::isfinite(value));
        repair_input_channel(token, channel, value);
        last_channel = static_cast<int32_t>(channel);
      }
    }
    MLLM_RT_ASSERT_EQ(dense_index, dense_count);
  }
}

}  // namespace mllm::cpu
