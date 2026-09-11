// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "mllm/core/BaseOp.hpp"

namespace mllm::aops {

// Grouped-query attention variants over native KV-head history. They differ in
// reduction order and in the shapes they accept, so the variant is part of the
// operation contract: callers bound to an exact generation-token oracle cannot
// be migrated between them silently.
enum class GroupedQueryAttentionImplementation : int32_t {
  // Any query length. Masks per query position and accumulates in the order
  // established by the eager reference.
  kDirectStrided = 0,
  // Single query position only. Uses the dedicated decode kernel, which is
  // faster but reduces in a different order.
  kDecodeNativeKV = 1,
};

inline const char* groupedQueryAttentionImplementation2Str(GroupedQueryAttentionImplementation implementation) {
  switch (implementation) {
    case GroupedQueryAttentionImplementation::kDirectStrided: return "DirectStrided";
    case GroupedQueryAttentionImplementation::kDecodeNativeKV: return "DecodeNativeKV";
  }
  return "Unknown";
}

inline GroupedQueryAttentionImplementation str2GroupedQueryAttentionImplementation(const std::string& value) {
  if (value == "DirectStrided") return GroupedQueryAttentionImplementation::kDirectStrided;
  if (value == "DecodeNativeKV") return GroupedQueryAttentionImplementation::kDecodeNativeKV;
  throw std::invalid_argument("Unknown GroupedQueryAttention implementation: " + value);
}

struct GroupedQueryAttentionOpOptions : public BaseOpOptions<GroupedQueryAttentionOpOptions> {
  GroupedQueryAttentionImplementation implementation = GroupedQueryAttentionImplementation::kDirectStrided;
  // Zero: full causal history. Positive: at most this many keys, including self.
  // Queries align with the last Sq positions of the chronological KV inputs.
  int32_t sliding_window = 0;
};

class GroupedQueryAttentionOp : public BaseOp {
 public:
  explicit GroupedQueryAttentionOp(const GroupedQueryAttentionOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;
  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const GroupedQueryAttentionOpOptions& options() const { return options_; }

 protected:
  GroupedQueryAttentionOpOptions options_;
};

}  // namespace mllm::aops
