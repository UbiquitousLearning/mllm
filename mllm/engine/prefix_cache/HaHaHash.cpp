// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <bit>
#include <cstring>
#include <xxHash/xxhash.h>

#include "mllm/utils/Common.hpp"
#include "mllm/engine/prefix_cache/HaHaHash.hpp"

namespace mllm::prefix_cache::hash {

namespace MLLM_ANONYMOUS_NAMESPACE {

inline void write_be64(unsigned char* dst, uint64_t v) {
  if constexpr (std::endian::native == std::endian::little) v = __builtin_bswap64(v);  // GCC/Clang can use this extention.
  std::memcpy(dst, &v, sizeof(v));
}

inline void write_be32(unsigned char* dst, uint32_t v) {
  if constexpr (std::endian::native == std::endian::little) v = __builtin_bswap32(v);
  std::memcpy(dst, &v, sizeof(v));
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

XXH64_hash_t ZenFSHashCode::hash() const noexcept {
  unsigned char buf[sizeof(uint64_t) * 2 + sizeof(uint32_t)];
  size_t off = 0;
  write_be64(buf + off, parent_hash_code);
  off += sizeof(uint64_t);
  write_be64(buf + off, content_hash_code);
  off += sizeof(uint64_t);
  write_be32(buf + off, extra_hash_code);
  return XXH64(buf, sizeof(buf), 0);
}

bool ZenFSHashCode::operator==(const ZenFSHashCode& o) const noexcept {
  return parent_hash_code == o.parent_hash_code && content_hash_code == o.content_hash_code
         && extra_hash_code == o.extra_hash_code;
}

XXH64_hash_t hashBytes(const void* data, size_t size) noexcept { return XXH64(data, size, 0); }

XXH64_hash_t hashBytes(const std::vector<uint8_t>& data) noexcept { return XXH64(data.data(), data.size(), 0); }

XXH64_hash_t hashBytes(const std::string& data) noexcept { return XXH64(data.data(), data.size(), 0); }
}  // namespace mllm::prefix_cache::hash
