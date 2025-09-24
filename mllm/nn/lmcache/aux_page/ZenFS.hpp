// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// ZenFS is a file system for storing Paged KV Cache
// The KV Cache will be stored in memory and in disk
// ZenFS will keep top 4K pages in memory and left others
// in disk. The other blocks can be r/w through mmap.

#pragma once

#include <string>
#include <optional>
#include <unordered_map>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/MappedFile.hpp"
#include "mllm/nn/lmcache/aux_page/TLB.hpp"

namespace mllm::nn::aux_page {

enum class ZenFSBlobType {
  kInMemory = 0,
  kOnDiskMMAP = 1,
};

struct ZenFSHeader {
  size_t blob_bits_size = 20;  // 2^20 blobs
  size_t page_bits_size = 7;   // 128 pages per blob
  size_t lane_bits_size = 5;   // 32 token per page
  std::string working_dir;
};

struct ZenFSBlob {
  Tensor data = Tensor::nil();
  ZenFSBlobType type = ZenFSBlobType::kInMemory;
  std::optional<MappedFile::ptr_t> mmap_file = std::nullopt;
};

class ZenFileSystem {
 public:
  ZenFileSystem() = default;

  vp_blob_addr_t malloc(size_t size);

  void free(vp_blob_addr_t addr);

  void* access(vp_blob_addr_t addr);

  void store();

  void recover();

 private:
  void _storeBlob0();

  void _allocNewBlobAndMMAP();

  void _initializeWorkSpace();

  ZenFSHeader header_;
  std::unordered_map<vp_blob_addr_t, ZenFSBlob> blob_;
};

}  // namespace mllm::nn::aux_page
