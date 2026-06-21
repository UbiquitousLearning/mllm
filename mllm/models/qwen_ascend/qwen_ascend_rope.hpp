// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::models::qwen_ascend {

inline auto makeLocalRoPEPositionIds(int batch_size, int seq_len) -> Tensor {
  auto rope_pos_ids = Tensor::empty({batch_size, seq_len}, kInt32, kCPU).alloc();
  auto* ptr = rope_pos_ids.ptr<int32_t>();
  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      ptr[b * seq_len + s] = s;
    }
  }
  return rope_pos_ids.to(kAscend);
}

inline auto makeRoPEInvFreq(int output_dim, float rope_theta) -> Tensor {
  auto inv_freq = Tensor::empty({output_dim / 2}, kFloat32, kCPU).alloc();
  auto inv_freq_ptr = inv_freq.ptr<float>();
  for (int i = 0; i < output_dim / 2; i++) {
    inv_freq_ptr[i] = 1.0f / std::pow(rope_theta, 2.0f * i / output_dim);
  }
  return inv_freq;
}

inline auto makeRotaryPosEmbedding(Tensor& position_ids, const Tensor& inv_freq,
                                   float attention_scaling = 1.0f) -> std::pair<Tensor, Tensor> {
  auto batch_size = position_ids.shape()[0];
  auto seq_len = position_ids.shape()[1];
  auto inv_freq_len = inv_freq.shape()[0];
  auto dim = inv_freq_len * 2;

  auto freqs = Tensor::empty({batch_size, seq_len, inv_freq_len}, kFloat32, kCPU).alloc();
  auto freqs_ptr = freqs.ptr<float>();
  auto position_ids_ptr = position_ids.ptr<int64_t>();
  auto inv_freq_ptr = inv_freq.ptr<float>();

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      auto pos = position_ids_ptr[b * seq_len + s];
      for (int d = 0; d < inv_freq_len; ++d) {
        freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d] = static_cast<float>(pos) * inv_freq_ptr[d];
      }
    }
  }

  auto sin_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto cos_emb = Tensor::empty({batch_size, seq_len, dim}, kFloat32, kCPU).alloc();
  auto sin_ptr = sin_emb.ptr<float>();
  auto cos_ptr = cos_emb.ptr<float>();

  for (int b = 0; b < batch_size; ++b) {
    for (int s = 0; s < seq_len; ++s) {
      for (int d = 0; d < inv_freq_len; ++d) {
        auto freq = freqs_ptr[b * seq_len * inv_freq_len + s * inv_freq_len + d];
        auto sin_val = std::sin(freq) * attention_scaling;
        auto cos_val = std::cos(freq) * attention_scaling;

        sin_ptr[b * seq_len * dim + s * dim + d] = sin_val;
        sin_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = sin_val;
        cos_ptr[b * seq_len * dim + s * dim + d] = cos_val;
        cos_ptr[b * seq_len * dim + s * dim + d + inv_freq_len] = cos_val;
      }
    }
  }

  return {sin_emb, cos_emb};
}

class QwenAscendRoPECache final {
 public:
  void clear() {
    cached_max_seq_len_ = 0;
    cached_sin_emb_ = Tensor::nil();
    cached_cos_emb_ = Tensor::nil();
  }

  std::pair<Tensor, Tensor> getEmbeddings(const Tensor& position_ids,
                                          const Tensor& inv_freq,
                                          int32_t max_cache_length) {
    auto batch_size = position_ids.shape()[0];
    auto seq_len = position_ids.shape()[1];

    auto pos_ptr = position_ids.ptr<int64_t>();
    int64_t max_pos = pos_ptr[0];
    for (int i = 1; i < batch_size * seq_len; ++i) {
      if (pos_ptr[i] > max_pos) {
        max_pos = pos_ptr[i];
      }
    }

    if (cached_max_seq_len_ <= max_pos || cached_sin_emb_.isNil() || cached_cos_emb_.isNil()) {
      int cache_size = std::min(static_cast<int>(max_pos + 1) * 2, static_cast<int>(max_cache_length));

      auto cache_position_ids = Tensor::empty({1, cache_size}, kInt64, kCPU).alloc();
      auto cache_pos_ptr = cache_position_ids.ptr<int64_t>();
      for (int i = 0; i < cache_size; ++i) {
        cache_pos_ptr[i] = i;
      }

      auto [sin_cpu, cos_cpu] = makeRotaryPosEmbedding(cache_position_ids, inv_freq, 1.0f);

      cached_sin_emb_ = sin_cpu.to(kFloat16).to(kAscend);
      cached_cos_emb_ = cos_cpu.to(kFloat16).to(kAscend);
      cached_max_seq_len_ = cache_size;
    }

    if (seq_len > 1 && pos_ptr[0] == 0) {
      bool is_contiguous = true;
      for (int i = 1; i < seq_len; ++i) {
        if (pos_ptr[i] != i) {
          is_contiguous = false;
          break;
        }
      }

      if (is_contiguous) {
        auto sin_slice = cached_sin_emb_[{kAll, {kAll, seq_len}, kAll}];
        auto cos_slice = cached_cos_emb_[{kAll, {kAll, seq_len}, kAll}];

        if (batch_size > 1) {
          sin_slice = sin_slice.repeat(batch_size, 0);
          cos_slice = cos_slice.repeat(batch_size, 0);
        }

        return {sin_slice, cos_slice};
      }
    }

    auto dim = cached_sin_emb_.shape()[2];
    auto sin_result = Tensor::empty({batch_size, seq_len, dim}, kFloat16, kAscend).alloc();
    auto cos_result = Tensor::empty({batch_size, seq_len, dim}, kFloat16, kAscend).alloc();

    for (int b = 0; b < batch_size; ++b) {
      for (int s = 0; s < seq_len; ++s) {
        int64_t pos = pos_ptr[b * seq_len + s];
        auto sin_pos = cached_sin_emb_[{kAll, {static_cast<int32_t>(pos), static_cast<int32_t>(pos + 1)}, kAll}];
        auto cos_pos = cached_cos_emb_[{kAll, {static_cast<int32_t>(pos), static_cast<int32_t>(pos + 1)}, kAll}];

        const size_t copy_size = dim * sizeof(mllm_fp16_t);
        auto ret = aclrtMemcpy(
            static_cast<char*>(sin_result.ptr<void>()) + (b * seq_len + s) * copy_size,
            copy_size,
            sin_pos.ptr<void>(),
            copy_size,
            ACL_MEMCPY_DEVICE_TO_DEVICE);
        MLLM_ACL_CHECK(ret);

        ret = aclrtMemcpy(
            static_cast<char*>(cos_result.ptr<void>()) + (b * seq_len + s) * copy_size,
            copy_size,
            cos_pos.ptr<void>(),
            copy_size,
            ACL_MEMCPY_DEVICE_TO_DEVICE);
        MLLM_ACL_CHECK(ret);
      }
    }

    mllm::ascend::syncGlobalAtbStream();
    return {sin_result, cos_result};
  }

 private:
  Tensor cached_sin_emb_;
  Tensor cached_cos_emb_;
  int cached_max_seq_len_{0};
};

}  // namespace mllm::models::qwen_ascend
