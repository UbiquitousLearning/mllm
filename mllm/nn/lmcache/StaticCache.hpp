// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/SlicePrimitives.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::nn {

class AbstractStaticCache {
 public:
  AbstractStaticCache() = default;
  [[nodiscard]] virtual int32_t getCurrentSeqCnt(int32_t layer_idx) const { return -1; };

  virtual void setCurrentSeqCnt(int32_t seq) {};

  [[nodiscard]] virtual int32_t getLayerNums() const { return 0; }

  virtual std::array<Tensor, 2> updateKVCache(int32_t layer_idx, Tensor k, Tensor v) {  // NOLINT
    return {Tensor::nil(), Tensor::nil()};
  }
};

class StaticCache : public AbstractStaticCache {
  friend class SubStaticCache;

 public:
  ~StaticCache() = default;

  StaticCache() = default;

  StaticCache(int32_t max_cache_length, int32_t layer_nums, int32_t q_heads, int32_t kv_heads, int32_t kv_dims,
              DataTypes k_dtype, DataTypes v_dtype, DeviceTypes device_type = kCPU, bool use_fa2 = true);

  void setCurrentSeqCnt(int32_t seq) override {
    for (int32_t layer_idx = 0; layer_idx < layer_nums_; ++layer_idx) { current_seq_cnt_[layer_idx] = seq; }
  }

  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const override;

  [[nodiscard]] int32_t getLayerNums() const override { return layer_nums_; }

  std::array<Tensor, 2> updateKVCache(int32_t layer_idx, Tensor k, Tensor v) override;

  [[nodiscard]] inline Tensor getKCache(int32_t layer_idx) const { return k_cache_[layer_idx]; };

  [[nodiscard]] inline Tensor getVCache(int32_t layer_idx) const { return v_cache_[layer_idx]; };

 private:
  DeviceTypes device_type_;
  DataTypes k_dtype_;
  DataTypes v_dtype_;
  int32_t max_cache_length_;
  int32_t layer_nums_;
  int32_t q_heads_;
  int32_t kv_heads_;
  int32_t kv_dims_;
  bool use_fa2_;

  std::vector<Tensor> k_cache_;
  std::vector<Tensor> v_cache_;
  std::vector<int32_t> current_seq_cnt_;
};

class SubStaticCache : public AbstractStaticCache {
 public:
  SubStaticCache(StaticCache& cache, int32_t start_idx, int32_t len) : cache_(cache), start_idx_(start_idx), len_(len) {
    int32_t layers = cache_.getLayerNums();
    sub_k_cache_.resize(layers);
    sub_v_cache_.resize(layers);
    current_sub_seq_cnt_.resize(layers, len_);

    sub_max_cache_length_ = cache_.max_cache_length_ - start_idx_;

    MLLM_RT_ASSERT(!cache_.use_fa2_);  // SubCache only supports non-fa2 mode for now

    for (int32_t i = 0; i < layers; i++) {
      // k_cache_ shape: [batch, kv_heads, seq_len, kv_dim]
      sub_k_cache_[i] = cache_.getKCache(i)[{kAll, kAll, {start_idx_, kAll}, kAll}];
      sub_v_cache_[i] = cache_.getVCache(i)[{kAll, kAll, {start_idx_, kAll}, kAll}];
    }
  }

  [[nodiscard]] int32_t getCurrentSeqCnt(int32_t layer_idx) const override;

  [[nodiscard]] int32_t getLayerNums() const override { return cache_.getLayerNums(); }

  std::array<Tensor, 2> updateKVCache(int32_t layer_idx, Tensor k, Tensor v) override;

 private:
  StaticCache& cache_;

  std::vector<Tensor> sub_k_cache_;
  std::vector<Tensor> sub_v_cache_;
  std::vector<int32_t> current_sub_seq_cnt_;
  int32_t start_idx_;
  int32_t len_;
  int32_t sub_max_cache_length_;  // max_cache_length(parent) - start_idx
};

}  // namespace mllm::nn
