// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/prefix_cache/TLB.hpp"

namespace mllm::prefix_cache {
vp_blob_addr_t getBlobAddr(vp_addr_t addr, size_t page_bits, size_t lane_bits) {
  size_t blob_bits = sizeof(vp_addr_t) - page_bits - lane_bits;
  return addr >> (page_bits + lane_bits);
}

vp_page_addr_t getPageAddr(vp_addr_t addr, size_t page_bits, size_t lane_bits) {
  uint64_t mask = (1ULL << page_bits) - 1;
  return (addr >> lane_bits) & mask;
}

vp_lane_addr_t getLaneAddr(vp_addr_t addr, size_t page_bits, size_t lane_bits) {
  uint64_t mask = (1ULL << lane_bits) - 1;
  return addr & mask;
}

void TLB::insert(vp_addr_t addr, char* data) { addr_space_.emplace(addr, data); }

void TLB::remove(vp_addr_t addr) { addr_space_.erase(addr); }

char* TLB::lookup(vp_addr_t addr) {
  auto it = addr_space_.find(addr);
  if (it != addr_space_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

}  // namespace mllm::prefix_cache
