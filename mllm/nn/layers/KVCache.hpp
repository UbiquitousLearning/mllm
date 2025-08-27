// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/KVCacheOp.hpp"

namespace mllm::nn {

class KVCache : public Layer {
 public:
  KVCache();

  explicit KVCache(const aops::KVCacheOpOptions& options);

  KVCache(int32_t layer_idx, int32_t q_head, int32_t kv_head, int32_t head_dim, bool use_fa2 = true);

  void setLayerIndex(int32_t layer_idx);

  MLLM_LAYER_ANY_INPUTS_2_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
