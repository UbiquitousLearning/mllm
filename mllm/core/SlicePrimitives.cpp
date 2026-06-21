// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/SlicePrimitives.hpp"

namespace mllm {

SliceIndicesPair::SliceIndicesPair(int32_t v) : start_(v), end_(v + 1) {
  if (v == kAll) {
    start_ = kAll;
    end_ = kAll;
  }
}

SliceIndicesPair::SliceIndicesPair(int32_t start, int32_t end, int32_t step) : start_(start), end_(end), step_(step) {}

}  // namespace mllm