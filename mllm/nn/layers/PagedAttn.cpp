// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/core/aops/PagedAttnOp.hpp"
#include "mllm/nn/layers/PagedAttn.hpp"

namespace mllm::nn {

PagedAttn::PagedAttn() : Layer(OpTypes::kPagedAttn, aops::PagedAttnOpOptions{}) {}

PagedAttn::PagedAttn(const aops::PagedAttnOpOptions& options) : Layer(OpTypes::kPagedAttn, options) {}

PagedAttn::PagedAttn(int32_t head_repeat_times, bool high_precision_exp, bool fuse_rope, bool need_attn_weights,
                     aops::PagedAttnImplType impl_type)
    : Layer(OpTypes::kPagedAttn, aops::PagedAttnOpOptions{.head_repeat_times = head_repeat_times,
                                                          .high_precision_exp = high_precision_exp,
                                                          .fuse_rope = fuse_rope,
                                                          .need_attn_weights = need_attn_weights,
                                                          .impl_type = impl_type}) {}

}  // namespace mllm::nn
