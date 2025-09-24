// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

//   _   _    _      _   _    _      _   _    _    ____  _   _
//  | | | |  / \    | | | |  / \    | | | |  / \  / ___|| | | |
//  | |_| | / _ \   | |_| | / _ \   | |_| | / _ \ \___ \| |_| |
//  |  _  |/ ___ \  |  _  |/ ___ \  |  _  |/ ___ \ ___) |  _  |
//  |_| |_/_/   \_\ |_| |_/_/   \_\ |_| |_/_/   \_\____/|_| |_|

#include <xxHash/xxhash.h>

namespace mllm::nn::aux_page {

// One Hash Code for one page
class ZenFSHashCode {
 public:
  XXH64_hash_t parent_hash_code;
  XXH64_hash_t content_hash_code;
  XXH32_hash_t extra_hash_code;

  [[nodiscard]] XXH64_hash_t hash() const;
};

}  // namespace mllm::nn::aux_page
