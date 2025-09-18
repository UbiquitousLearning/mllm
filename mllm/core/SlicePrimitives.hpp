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

// Helper class for comma operator to enable [1,2,3] syntax
class SliceIndicesBuilder {
 public:
  // NOLINT for intentional implicit conversion
  SliceIndicesBuilder(int32_t first_index) {  // NOLINT(google-explicit-constructor)
    indices_.emplace_back(first_index);
  }

  // NOLINT for intentional implicit conversion
  SliceIndicesBuilder(const SliceIndicesPair& first_pair) {  // NOLINT(google-explicit-constructor)
    indices_.emplace_back(first_pair);
  }

  // operator, to chain multiple indices
  SliceIndicesBuilder operator,(int32_t index) && {
    indices_.emplace_back(index);
    return std::move(*this);
  }

  SliceIndicesBuilder operator,(const SliceIndicesPair& pair) && {
    indices_.emplace_back(pair);
    return std::move(*this);
  }

  // Implicit conversion to SliceIndices - intentional for syntax sugar
  operator SliceIndices() const {  // NOLINT(google-explicit-constructor)
    return indices_;
  }

 private:
  SliceIndices indices_;
};

// Helper function to start the builder chain
inline SliceIndicesBuilder make_slice(int32_t index) { return {index}; }

inline SliceIndicesBuilder make_slice(const SliceIndicesPair& pair) { return {pair}; }

}  // namespace mllm
