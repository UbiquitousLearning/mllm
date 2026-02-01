// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include <cstdint>

namespace mllm {

enum SliceIndexPlaceHolder : int32_t {
  kSliceIndexPlaceHolder_Start = 0x7FFFFFF0,
  kAll,  // 0x7FFFFFF1
  kSliceIndexPlaceHolder_End,
};

struct SliceIndicesPair {
  SliceIndicesPair() = default;
  SliceIndicesPair(int32_t v);  // NOLINT(google-explicit-constructor)
  SliceIndicesPair(int32_t start, int32_t end, int32_t step = 1);

  int32_t start_ = kAll;
  int32_t end_ = kAll;
  int32_t step_ = 1;
};

using SliceIndices = std::vector<SliceIndicesPair>;

}  // namespace mllm
