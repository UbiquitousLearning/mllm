// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/nn/Layer.hpp"
#include "mllm/core/aops/PagedAttnOp.hpp"

namespace mllm::nn {

class PagedAttn : public Layer {
 public:
  PagedAttn();

  explicit PagedAttn(const aops::PagedAttnOpOptions& options);

  explicit PagedAttn(int32_t head_repeat_times, bool high_precision_exp = false, bool fuse_rope = false,
                     bool need_attn_weights = false, aops::PagedAttnImplType impl_type = aops::PagedAttnImplType::kAllFp32);

  MLLM_LAYER_ANY_INPUTS_2_OUTPUTS_FORWARD
};

}  // namespace mllm::nn
