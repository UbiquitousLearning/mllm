// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct KimiDeltaAttentionOpOptions : public BaseOpOptions<KimiDeltaAttentionOpOptions> {
  bool safe_gate = true;
  float lower_bound = -5.0F;
  bool state_inplace = false;
};

class KimiDeltaAttentionOp : public BaseOp {
 public:
  explicit KimiDeltaAttentionOp(const KimiDeltaAttentionOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const KimiDeltaAttentionOpOptions& options() const { return options_; }

 protected:
  KimiDeltaAttentionOpOptions options_;
};

}  // namespace mllm::aops
