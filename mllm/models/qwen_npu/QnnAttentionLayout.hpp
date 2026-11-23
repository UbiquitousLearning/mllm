// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

namespace mllm::models::qwen_npu {

template <typename Output, typename WriteRow>
inline void writeQnnAttentionRows(const float* source, Output* destination, size_t batches, size_t heads, size_t tokens,
                                  size_t head_dim, WriteRow write_row) {
  const size_t head_stride = tokens * head_dim;
  const size_t batch_stride = heads * head_stride;
  for (size_t batch = 0; batch < batches; ++batch) {
    const auto* batch_source = source + batch * batch_stride;
    for (size_t token = 0; token < tokens; ++token) {
      for (size_t head = 0; head < heads; ++head) {
        write_row(batch_source + head * head_stride + token * head_dim, destination, head_dim);
        destination += head_dim;
      }
    }
  }
}

}  // namespace mllm::models::qwen_npu
