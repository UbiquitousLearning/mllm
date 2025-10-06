// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "mllm/engine/prefix_cache/ZenFS.hpp"

namespace mllm::prefix_cache {

///< An abstraction of the token level allocator
class TokenLevelAllocator {
 public:
  virtual vp_blob_addr_t alloc() = 0;
  virtual void free(vp_blob_addr_t addr) = 0;
};

class CpuTokenLevelAllocator final : public TokenLevelAllocator {
 public:
  vp_blob_addr_t alloc() override;
  void free(vp_blob_addr_t addr) override;

 private:
  ZenFileSystem zen_fs_;
};

}  // namespace mllm::prefix_cache
