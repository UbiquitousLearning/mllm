// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// ZenFS is a file system for storing Paged KV Cache and history messages.
// The KV Cache will be stored in memory and in disk
// ZenFS will keep top 4K pages in memory and left others
// in disk. The other blocks can be r/w through mmap.

#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <system_error>

// MMAP Stuff
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/engine/prefix_cache/TLB.hpp"

namespace mllm::prefix_cache {

enum class ZenFSMMAPMode : int32_t {
  kReadOnly,
  kReadWrite,
  kCopyOnWrite,
  kAnonymous,
};

// From POSIX:
// The implementation performs mapping operations over whole pages. Thus, addr shall be page-aligned.
//
// The page size is always 4KiB Aligned.
class ZenFSBlobMMAPFile {
 public:
  using ptr_t = std::shared_ptr<ZenFSBlobMMAPFile>;

  static ptr_t create(size_t size, ZenFSMMAPMode mode, const std::string& fpath, std::error_code& ec);

  ~ZenFSBlobMMAPFile();

  // Copy is not allowed, move is ok.
  ZenFSBlobMMAPFile(const ZenFSBlobMMAPFile&) = delete;
  ZenFSBlobMMAPFile& operator=(const ZenFSBlobMMAPFile&) = delete;
  ZenFSBlobMMAPFile(ZenFSBlobMMAPFile&&) = default;
  ZenFSBlobMMAPFile& operator=(ZenFSBlobMMAPFile&&) = default;

  [[nodiscard]] inline uint8_t* data() const { return addr_; }

  [[nodiscard]] inline size_t size() const { return size_; }

  // Hints
  void adviseSequential();
  void adviseRandom();
  void adviseWillNeed();
  void adviseDontNeed();
  void adviseOffline();  // Linux: MADV_OFFLINE

  // Primitives
  void prefetch(size_t offset, size_t len) const;
  void purge(size_t offset, size_t len);
  void sync() const;

 private:
  ZenFSBlobMMAPFile() = default;

  bool initAnonymous(size_t size, std::error_code& ec);

  bool initFileBacked(size_t size, ZenFSMMAPMode mode, const std::string& fpath, std::error_code& ec);

#if defined(_WIN32)
  void cleanupWindows();
  HANDLE file_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_ = nullptr;
#else
  void cleanupPosix();
  int fd_ = -1;
#endif
  uint8_t* addr_ = nullptr;
  size_t size_ = 0;
};

enum class ZenFSBlobMMapType : int32_t {
  kAnonymous = 0,
  kFile = 1,
};

enum class ZenFSBlobType : int32_t {
  kInMemory = 0,
  kOnDiskMMAP = 1,
  kZSTDCompressed = 2,
};

struct ZenFSBlob {
  Tensor data = Tensor::nil();
  ZenFSBlobType type = ZenFSBlobType::kInMemory;
  std::optional<ZenFSBlobMMAPFile::ptr_t> mmap_file = std::nullopt;
  std::vector<uint64_t> free_lanes_bits_map;  // 1024 bits, 16 uint64_t most likely.
};

struct ZenFileSystemOptions {
  bool record = false;
  std::string working_dir;
  size_t blob_bits_size = 20;
  size_t page_bits = 7;  // 128 Pages per blob, 64 Page K, 64 Page V. K V K V K V.
  size_t lane_bits = 5;  // 32 token per page
  size_t per_k_token_ele = 1024;
  size_t per_v_token_ele = 1024;
  DataTypes k_dtype = kFloat16;
  DataTypes v_dtype = kFloat16;
  ZenFSBlobMMapType mmap_type = ZenFSBlobMMapType::kAnonymous;
};

class ZenFileSystem {
 public:
  ZenFileSystem() = default;

  vp_addr_t malloc();

  void free(vp_addr_t addr);

  char* access(vp_addr_t addr);

  void initialize(const ZenFileSystemOptions& options);

  void finalize();

  void recover(const std::string& working_dir);

  void hintsPrefetch(vp_addr_t token_lane_addr) const;

  void hintsPurge(vp_addr_t token_lane_addr);

 private:
  void _createBlobOnDisk();

  void _createBlobOnAnonymousFile();

  void _findFreeAddrInBlob(vp_blob_addr_t blob_addr, vp_addr_t* ret_addr);

  nlohmann::json index_;
  ZenFileSystemOptions options_;
  size_t per_kv_token_mem_size_;
  std::unordered_map<vp_blob_addr_t, ZenFSBlob> blob_;
};

}  // namespace mllm::prefix_cache
