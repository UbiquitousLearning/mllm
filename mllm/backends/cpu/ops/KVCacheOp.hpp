// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/aops/KVCacheOp.hpp"
#include "mllm/nn/lmcache/StaticCache.hpp"

namespace mllm::cpu {

class CPUKVCacheOp final : public aops::KVCacheOp {
 public:
  explicit CPUKVCacheOp(const aops::KVCacheOpOptions& options);

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void clearCache() override;

  void setCurrentSeqCnt(int32_t seq) override;

  int32_t getCurrentSeqCnt() const override;

  void setStaticCache(const nn::StaticCache* cache) override { shared_cache_ = const_cast<nn::StaticCache*>(cache); }
  const nn::StaticCache* getStaticCache() const override { return shared_cache_ ? shared_cache_ : &cache_; }

 private:
  nn::StaticCache cache_;
  nn::StaticCache* shared_cache_ = nullptr;
};

class CPUKVCacheOpFactory : public TypedOpFactory<OpTypes::kKVCache, aops::KVCacheOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::KVCacheOpOptions& options) override {
    return std::make_shared<CPUKVCacheOp>(options);
  }
};

}  // namespace mllm::cpu
