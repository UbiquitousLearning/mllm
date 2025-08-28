// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct KVCacheOpOptions : public BaseOpOptions<KVCacheOpOptions> {
  int32_t layer_idx = 0;
  int32_t q_head = 0;
  int32_t kv_head = 0;
  int32_t head_dim = 0;
  bool use_fa2 = true;
};

class KVCacheOp : public BaseOp {
 public:
  explicit KVCacheOp(const KVCacheOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setLayerIndex(int32_t layer_idx);

  inline const KVCacheOpOptions& options() const { return options_; }

 protected:
  KVCacheOpOptions options_;
};

}  // namespace mllm::aops