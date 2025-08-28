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

 private:
  nn::StaticCache cache_;
};

class CPUKVCacheOpFactory : public TypedOpFactory<OpTypes::kKVCache, aops::KVCacheOpOptions> {
 public:
  std::shared_ptr<BaseOp> createOpImpl(const aops::KVCacheOpOptions& options) override {
    return std::make_shared<CPUKVCacheOp>(options);
  }
};

}  // namespace mllm::cpu
