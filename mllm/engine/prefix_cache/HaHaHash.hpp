// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

//   _   _    _      _   _    _      _   _    _    ____  _   _
//  | | | |  / \    | | | |  / \    | | | |  / \  / ___|| | | |
//  | |_| | / _ \   | |_| | / _ \   | |_| | / _ \ \___ \| |_| |
//  |  _  |/ ___ \  |  _  |/ ___ \  |  _  |/ ___ \ ___) |  _  |
//  |_| |_/_/   \_\ |_| |_/_/   \_\ |_| |_/_/   \_\____/|_| |_|

#include <vector>
#include <string>
#include <xxHash/xxhash.h>

namespace mllm::prefix_cache::hash {

// One Hash Code for one page
class ZenFSHashCode {
 public:
  XXH64_hash_t parent_hash_code;
  XXH64_hash_t content_hash_code;
  XXH32_hash_t extra_hash_code;

  [[nodiscard]] XXH64_hash_t hash() const noexcept;

  bool operator==(const ZenFSHashCode& o) const noexcept;
};

XXH64_hash_t hashBytes(const void* data, size_t size) noexcept;

XXH64_hash_t hashBytes(const std::vector<uint8_t>& data) noexcept;

XXH64_hash_t hashBytes(const std::string& data) noexcept;

}  // namespace mllm::prefix_cache::hash
