// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

// 32bits Addr [ blob_bits_size, page_bits_size, lane_bits_size ]
//
// Normally set:  lane_bits_size to 5(32 tokens), 6(64 tokens), 7(128 tokens)
//                page_bits_size to 7(128 pages), 6(64 pages), 5(32 pages). 4K Tokens per page.
//
// Each 4K Tokens will be saved in a page. Indexing by blob id. The first page will always in memory
// And left will be on disk and callback through mmap.

#include <cstddef>
#include <cstdint>
#include <unordered_map>

#define INVALID_VP_ADDR 0xFFFFFFFF

namespace mllm::prefix_cache {

using vp_addr_t = uint32_t;
using vp_blob_addr_t = uint32_t;
using vp_page_addr_t = uint32_t;
using vp_lane_addr_t = uint32_t;

vp_blob_addr_t getBlobAddr(vp_addr_t addr, size_t page_bits, size_t lane_bits);

vp_page_addr_t getPageAddr(vp_addr_t addr, size_t page_bits, size_t lane_bits);

vp_lane_addr_t getLaneAddr(vp_addr_t addr, size_t page_bits, size_t lane_bits);

class TLB {
 public:
  void insert(vp_addr_t addr, char* data);

  void remove(vp_addr_t addr);

  char* lookup(vp_addr_t addr);

 private:
  std::unordered_map<vp_addr_t, char*> addr_space_;
};

}  // namespace mllm::prefix_cache
