// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once
#include <string_view>

namespace mllm::qnn {

inline constexpr std::string_view shadow_raw_graph_suffix = ".shadow_raw";

inline std::string_view shadowMainGraphName(std::string_view name) {
  if (name.ends_with(shadow_raw_graph_suffix)) { name.remove_suffix(shadow_raw_graph_suffix.size()); }
  return name;
}

}  // namespace mllm::qnn
