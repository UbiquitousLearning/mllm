// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <new>
#include <chrono>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <system_error>
#include <xxHash/xxhash.h>
#include <nlohmann/json.hpp>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/engine/prefix_cache/ZenFS.hpp"

namespace mllm::prefix_cache {

ZenFSBlobMMAPFile::ptr_t ZenFSBlobMMAPFile::create(size_t size, ZenFSMMAPMode mode, const std::string& fpath,
                                                   std::error_code& ec) {
  // Not throw std::bad_alloc but return nullptr
  auto ptr = std::shared_ptr<ZenFSBlobMMAPFile>(new (std::nothrow) ZenFSBlobMMAPFile);
  if (!ptr) {
    ec = std::make_error_code(std::errc::not_enough_memory);
    return nullptr;
  }
  bool ok = false;
  if (mode == ZenFSMMAPMode::kAnonymous) {
    ok = ptr->initAnonymous(size, ec);
  } else {
    ok = ptr->initFileBacked(size, mode, fpath, ec);
  }
  if (!ok) { ptr.reset(); }
  return ptr;
}

#if !defined(_WIN32)

bool ZenFSBlobMMAPFile::initAnonymous(size_t size, std::error_code& ec) {
  size_ = size;
  int flags = MAP_ANONYMOUS | MAP_PRIVATE;
  int prot = PROT_READ | PROT_WRITE;
  addr_ = static_cast<uint8_t*>(::mmap(nullptr, size, prot, flags, -1, 0));
  if (addr_ == MAP_FAILED) {
    ec = std::error_code(errno, std::system_category());
    addr_ = nullptr;
    return false;
  }
  return true;
}

bool ZenFSBlobMMAPFile::initFileBacked(size_t size, ZenFSMMAPMode mode, const std::string& fpath, std::error_code& ec) {
  size_ = size;
  std::vector<char> tmp(fpath.begin(), fpath.end());
  tmp.push_back('\0');
  fd_ = ::mkstemp(tmp.data());
  if (fd_ < 0) {
    ec = std::error_code(errno, std::system_category());
    return false;
  }
  ::unlink(tmp.data());  // File is deleted when fd_ is closed
  if (::ftruncate(fd_, static_cast<off_t>(size)) != 0) {
    ec = std::error_code(errno, std::system_category());
    ::close(fd_);
    fd_ = -1;
    return false;
  }

  int prot = 0;
  int flags = 0;
  switch (mode) {
    case ZenFSMMAPMode::kReadOnly: {
      prot = PROT_READ;
      flags = MAP_SHARED;
      break;
    }
    case ZenFSMMAPMode::kReadWrite: {
      prot = PROT_READ | PROT_WRITE;
      flags = MAP_SHARED;
      break;
    }
    case ZenFSMMAPMode::kCopyOnWrite: {
      prot = PROT_READ | PROT_WRITE;
      flags = MAP_PRIVATE;
      break;
    }
    default: {
      ec = std::make_error_code(std::errc::invalid_argument);
      return false;
    }
  }

  addr_ = static_cast<uint8_t*>(::mmap(nullptr, size, prot, flags, fd_, 0));
  if (addr_ == MAP_FAILED) {
    ec = std::error_code(errno, std::system_category());
    addr_ = nullptr;
    cleanupPosix();
    return false;
  }
  return true;
}

void ZenFSBlobMMAPFile::cleanupPosix() {
  if (addr_) {
    ::munmap(addr_, size_);
    addr_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

ZenFSBlobMMAPFile::~ZenFSBlobMMAPFile() { cleanupPosix(); }

void ZenFSBlobMMAPFile::adviseSequential() { ::posix_madvise(addr_, size_, POSIX_MADV_SEQUENTIAL); }

void ZenFSBlobMMAPFile::adviseRandom() { ::posix_madvise(addr_, size_, POSIX_MADV_RANDOM); }

void ZenFSBlobMMAPFile::adviseWillNeed() { ::posix_madvise(addr_, size_, POSIX_MADV_WILLNEED); }

void ZenFSBlobMMAPFile::adviseDontNeed() { ::posix_madvise(addr_, size_, POSIX_MADV_DONTNEED); }

void ZenFSBlobMMAPFile::adviseOffline() {
#ifdef MADV_OFFLINE
  ::madvise(addr_, size_, MADV_OFFLOAD);
#endif
}

void ZenFSBlobMMAPFile::prefetch(size_t offset, size_t len) const {
  if (offset + len > size_) return;
  uint8_t* start = addr_ + offset;
#if defined(_WIN32)
  WIN32_MEMORY_RANGE_ENTRY entry{start, len};
  ::PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#else
  ::madvise(start, len, MADV_WILLNEED);
#endif
}

void ZenFSBlobMMAPFile::purge(size_t offset, size_t len) {
  if (offset + len > size_) return;
  uint8_t* start = addr_ + offset;
#if defined(_WIN32)
  ::FlushViewOfFile(start, len);
  ::DiscardVirtualMemory(start, len);
#else
  ::msync(start, len, MS_SYNC);
  ::madvise(start, len, MADV_DONTNEED);
#endif
}

void ZenFSBlobMMAPFile::sync() const {
#if defined(_WIN32)
  if (addr_) { ::FlushViewOfFile(addr_, size_); }
#else
  if (addr_) { ::msync(addr_, size_, MS_SYNC); }
#endif
}

// -------------------- Windows --------------------
#else  // _WIN32
bool ZenFSBlobMMAPFile::initAnonymous(size_t size, std::error_code& ec) {
  size_ = size;
  wchar_t tempPath[MAX_PATH];
  wchar_t tempFile[MAX_PATH];
  ::GetTempPathW(MAX_PATH, tempPath);
  ::GetTempFileNameW(tempPath, L"zen", 0, tempFile);

  file_ = ::CreateFileW(tempFile, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                        FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
  if (file_ == INVALID_HANDLE_VALUE) {
    ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    return false;
  }
  LARGE_INTEGER li;
  li.QuadPart = static_cast<LONGLONG>(size);
  if (!::SetFilePointerEx(file_, li, nullptr, FILE_BEGIN) || !::SetEndOfFile(file_)) {
    ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    cleanupWindows();
    return false;
  }

  mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READWRITE, static_cast<DWORD>(size >> 32),
                                  static_cast<DWORD>(size & 0xFFFFFFFF), nullptr);
  if (!mapping_) {
    ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    cleanupWindows();
    return false;
  }

  DWORD flProtect = FILE_MAP_WRITE;
  addr_ = static_cast<uint8_t*>(::MapViewOfFile(mapping_, flProtect, 0, 0, size));
  if (!addr_) {
    ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    cleanupWindows();
    return false;
  }
  return true;
}

bool ZenFSBlobMMAPFile::initFileBacked(size_t size, ZenFSMMAPMode mode, const std::string& fpath, std::error_code& ec) {
  size_ = size;

  std::wstring wfpath(fpath.begin(), fpath.end());

  DWORD desiredAccess = 0;
  DWORD flProtect = 0;
  DWORD mapProtect = 0;

  switch (mode) {
    case ZenFSMMAPMode::kReadOnly:
      desiredAccess = GENERIC_READ;
      flProtect = PAGE_READONLY;
      mapProtect = FILE_MAP_READ;
      break;
    case ZenFSMMAPMode::kReadWrite:
      desiredAccess = GENERIC_READ | GENERIC_WRITE;
      flProtect = PAGE_READWRITE;
      mapProtect = FILE_MAP_WRITE;
      break;
    case ZenFSMMAPMode::kCopyOnWrite:
      desiredAccess = GENERIC_READ | GENERIC_WRITE;
      flProtect = PAGE_WRITECOPY;
      mapProtect = FILE_MAP_COPY;
      break;
    default: ec = std::make_error_code(std::errc::invalid_argument); return false;
  }

  file_ = ::CreateFileW(wfpath.c_str(), desiredAccess, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file_ == INVALID_HANDLE_VALUE) {
    ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    return false;
  }

  LARGE_INTEGER li;
  li.QuadPart = static_cast<LONGLONG>(size);
  if (!::SetFilePointerEx(file_, li, nullptr, FILE_BEGIN) || !::SetEndOfFile(file_)) {
    ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    cleanupWindows();
    return false;
  }

  mapping_ = ::CreateFileMappingW(file_, nullptr, flProtect, static_cast<DWORD>(size >> 32),
                                  static_cast<DWORD>(size & 0xFFFFFFFF), nullptr);
  if (!mapping_) {
    ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    cleanupWindows();
    return false;
  }

  addr_ = static_cast<uint8_t*>(::MapViewOfFile(mapping_, mapProtect, 0, 0, size));
  if (!addr_) {
    ec = std::error_code(static_cast<int>(::GetLastError()), std::system_category());
    cleanupWindows();
    return false;
  }
  return true;
}

void ZenFSBlobMMAPFile::cleanupWindows() {
  if (addr_) {
    ::UnmapViewOfFile(addr_);
    addr_ = nullptr;
  }
  if (mapping_) {
    ::CloseHandle(mapping_);
    mapping_ = nullptr;
  }
  if (file_ != INVALID_HANDLE_VALUE) {
    ::CloseHandle(file_);
    file_ = INVALID_HANDLE_VALUE;
  }
}

ZenFSBlobMMAPFile::~ZenFSBlobMMAPFile() { cleanupWindows(); }

void ZenFSBlobMMAPFile::adviseSequential() {
  WIN32_MEMORY_RANGE_ENTRY entry{addr_, size_};
  ::PrefetchVirtualMemory(::GetCurrentProcess(), 1, &entry, 0);
}

void ZenFSBlobMMAPFile::adviseRandom() { MLLM_EMPTY_SCOPE }

void ZenFSBlobMMAPFile::adviseWillNeed() {
  WIN32_MEMORY_RANGE_ENTRY entry{addr_, size_};
  ::PrefetchVirtualMemory(::GetCurrentProcess(), 1, &entry, 0);
}

void ZenFSBlobMMAPFile::adviseDontNeed() { ::DiscardVirtualMemory(addr_, size_); }

void ZenFSBlobMMAPFile::adviseOffline() { ::DiscardVirtualMemory(addr_, size_); }
#endif

void ZenFileSystem::initialize(const ZenFileSystemOptions& options) {
  options_ = options;
  per_kv_token_mem_size_ = bytesOfType(options.k_dtype) * options.per_k_token_ele / lanesOfType(options.k_dtype);

  // 1. Check if this working dirs exists
  if (options_.record) {
    if (!std::filesystem::exists(options.working_dir)) { std::filesystem::create_directory(options.working_dir); }
  }

  // 2. Create index.json instance
  auto ms =
      std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  index_["uuid"] = XXH64(&ms, sizeof(ms), 0);
  index_["working_dir"] = options.working_dir;
  index_["blob_bits_size"] = options.blob_bits_size;
  index_["lane_bits_size"] = options.lane_bits;
  index_["page_bits_size"] = options.page_bits;
  index_["per_k_token_ele"] = options.per_k_token_ele;
  index_["per_v_token_ele"] = options.per_v_token_ele;
  index_["k_dtype"] = static_cast<int>(options.k_dtype);
  index_["v_dtype"] = static_cast<int>(options.v_dtype);
  index_["mmap_type"] = static_cast<int>(options.mmap_type);

  // 3. Validate options
  if (options.per_k_token_ele != options.per_v_token_ele) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "per_k_token_ele != per_v_token_ele");
  }
  if (options.k_dtype != options.v_dtype) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "k_dtype != v_dtype"); }
  MLLM_RT_ASSERT(options.page_bits + options.lane_bits < 32);

  // X. Create the first block Tensor for KV
  {
    size_t total_bits = (1 << (options_.page_bits + options_.lane_bits));
    size_t uint64_count = (total_bits + 64 - 1) / 64;

    // The first blob is in memory
    blob_.insert({blob_.size(), ZenFSBlob{.data = Tensor::zeros({(1 << (options_.page_bits + options_.lane_bits)),
                                                                 (int32_t)options.per_k_token_ele},
                                                                options.k_dtype),
                                          .type = ZenFSBlobType::kInMemory,
                                          .mmap_file = std::nullopt,
                                          .free_lanes_bits_map = std::vector<uint64_t>(uint64_count, ~0ULL)}});
  }
}

void ZenFileSystem::finalize() {
  // 1. Write to index.json
  if (options_.record) {
    nlohmann::json blobs_info = nlohmann::json::array();
    for (const auto& [blob_id, blob] : blob_) {
      nlohmann::json blob_info;
      blob_info["id"] = blob_id;
      blob_info["type"] = static_cast<int>(blob.type);
      blob_info["data_shape"] = blob.data.shape();
      blob_info["data_dtype"] = blob.data.dtype();

      // Record if blob has mmap file
      if (blob.mmap_file.has_value()) {
        blob_info["has_mmap_file"] = true;
        blob_info["mmap_file_path"] = options_.working_dir + "/blob_" + std::to_string(blob_id) + ".kv";
      } else {
        blob_info["has_mmap_file"] = false;
      }

      blobs_info.push_back(blob_info);
    }
    index_["blobs"] = blobs_info;

    std::string index_file_path = options_.working_dir + "/index.json";
    std::ofstream index_file(index_file_path);
    if (index_file.is_open()) {
      index_file << index_ << "\n";
      index_file.close();
    }
  }

  // 2. Write first Tensor Blob into disk.
  // The first blob (blob id 0) should be written to disk
  auto it = blob_.find(0);
  if (it != blob_.end() && options_.record) {
    std::string blob_file_path = options_.working_dir + "/blob_0.kv";
    // Write the tensor data to disk
    auto& tensor = it->second.data;
    auto* data_ptr = tensor.ptr<char>();
    auto data_size = tensor.bytes();
    std::ofstream blob_file(blob_file_path, std::ios::binary);
    if (blob_file.is_open()) {
      blob_file.write(data_ptr, data_size);
      blob_file.close();
    }
  }

  // 3. Write back mmap file if some page is dirty
  for (auto& [blob_id, blob] : blob_) {
    if (blob.type == ZenFSBlobType::kOnDiskMMAP && blob.mmap_file.has_value()) {
      // Sync the mmap file to ensure all changes are written to disk
      blob.mmap_file.value()->sync();
    }
  }
}

void ZenFileSystem::recover(const std::string& working_dir) {
  // 1. Check if working directory exists
  if (!std::filesystem::exists(working_dir)) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Working directory does not exist"); }

  // 2. Check if index.json exists
  std::string index_file_path = working_dir + "/index.json";
  if (!std::filesystem::exists(index_file_path)) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "index.json not found in working directory");
  }

  // 3. Read and parse index.json
  std::ifstream index_file(index_file_path);
  if (!index_file.is_open()) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Failed to open index.json"); }

  nlohmann::json index;
  try {
    index_file >> index;
    index_file.close();
  } catch (const std::exception& e) {
    MLLM_ERROR_EXIT(ExitCode::kCoreError, "{}", "Failed to parse index.json: " + std::string(e.what()));
  }

  // 4. Recover options from index
  options_.working_dir = working_dir;
  options_.blob_bits_size = index.value("blob_bits_size", 20);
  options_.lane_bits = index.value("lane_bits_size", 5);
  options_.page_bits = index.value("page_bits_size", 7);
  options_.per_k_token_ele = index.value("per_k_token_ele", 1024);
  options_.per_v_token_ele = index.value("per_v_token_ele", 1024);

  // Parse data types
  auto k_dtype_str = index.value("k_dtype", kFloat16);
  options_.k_dtype = static_cast<DataTypes>(k_dtype_str);
  auto v_dtype_str = index.value("v_dtype", kFloat16);
  options_.v_dtype = static_cast<DataTypes>(v_dtype_str);

  options_.mmap_type = static_cast<ZenFSBlobMMapType>(index.value("mmap_type", 0));
  options_.record = true;

  // 5. Set per token memory size
  per_kv_token_mem_size_ = bytesOfType(options_.k_dtype) * options_.per_k_token_ele / lanesOfType(options_.k_dtype);

  // 6. Recreate blobs from index info
  if (index.contains("blobs") && index["blobs"].is_array()) {
    for (const auto& blob_info : index["blobs"]) {
      uint32_t blob_id = blob_info.value("id", 0);
      ZenFSBlobType blob_type = static_cast<ZenFSBlobType>(blob_info.value("type", 0));

      // Parse shape
      TensorViewImpl::shape_t shape;
      if (blob_info.contains("data_shape") && blob_info["data_shape"].is_array()) {
        auto shape_array = blob_info["data_shape"];
        for (auto& dim : shape_array) { shape.push_back(dim.get<int32_t>()); }
      }

      // Parse dtype
      DataTypes dtype = kFloat16;
      if (blob_info.contains("data_dtype")) { dtype = static_cast<DataTypes>(blob_info.value("data_dtype", kFloat16)); }

      ZenFSBlob blob;
      blob.type = blob_type;

      switch (blob_type) {
        case ZenFSBlobType::kInMemory: {
          // For in-memory blob, create tensor directly
          blob.data = Tensor::zeros(shape, dtype);
          break;
        }
        case ZenFSBlobType::kOnDiskMMAP: {
          // For mmap blob, need to map file
          if (blob_info.value("has_mmap_file", false)) {
            std::string mmap_file_path = blob_info.value("mmap_file_path", "");
            if (!mmap_file_path.empty() && std::filesystem::exists(mmap_file_path)) {
              size_t blob_size = std::filesystem::file_size(mmap_file_path);

              std::error_code ec;
              auto mmap_file = ZenFSBlobMMAPFile::create(blob_size, ZenFSMMAPMode::kReadWrite, mmap_file_path, ec);

              if (!ec) {
                // Create tensor on MMAP
                auto s = TensorStorage::create(shape, dtype, kCPU);
                auto t = TensorViewImpl::create(shape, s);
                s->name_ = std::to_string(blob_id);
                s->ptr_ = mmap_file->data();
                s->mem_type_ = kParamsMMAP;
                blob.data = Tensor(t);
                blob.mmap_file = mmap_file;
              }
            }
          }
          break;
        }
        case ZenFSBlobType::kZSTDCompressed:
          // TODO: Handle compressed blobs if needed
          break;
      }

      // Initialize free lanes bitmap
      size_t total_bits = (1 << (options_.page_bits + options_.lane_bits));
      size_t uint64_count = (total_bits + 64 - 1) / 64;
      blob.free_lanes_bits_map = std::vector<uint64_t>(uint64_count, ~0ULL);

      // Insert blob into map
      blob_.insert({blob_id, std::move(blob)});
    }
  }

  // 7. Load blob_0.kv file if exists
  std::string blob0_file_path = working_dir + "/blob_0.kv";
  if (std::filesystem::exists(blob0_file_path)) {
    auto it = blob_.find(0);
    if (it != blob_.end() && it->second.type == ZenFSBlobType::kInMemory) {
      std::ifstream blob0_file(blob0_file_path, std::ios::binary);
      if (blob0_file.is_open()) {
        auto& tensor = it->second.data;
        auto* data_ptr = tensor.ptr<char>();
        auto data_size = tensor.bytes();

        if (data_ptr) { blob0_file.read(data_ptr, data_size); }
        blob0_file.close();
      }
    }
  }
}

void ZenFileSystem::hintsPrefetch(vp_addr_t token_lane_addr) const {
  // 1. Find which blob we are.
  vp_blob_addr_t blob_addr = getBlobAddr(token_lane_addr, options_.page_bits, options_.lane_bits);

  // 2. If this blob in mmap file, we need to prefetch.
  auto it = blob_.find(blob_addr);
  if (it != blob_.end() && it->second.type == ZenFSBlobType::kOnDiskMMAP && it->second.mmap_file.has_value()) {
    // Calculate the offset and length for prefetch
    // Each lane represents a token, and we need to prefetch based on the token lane address
    vp_page_addr_t page_addr = getPageAddr(token_lane_addr, options_.page_bits, options_.lane_bits);
    vp_lane_addr_t lane_addr = getLaneAddr(token_lane_addr, options_.page_bits, options_.lane_bits);

    // Calculate offset: page offset + lane offset
    size_t page_offset = page_addr * per_kv_token_mem_size_;
    size_t lane_offset = lane_addr * per_kv_token_mem_size_;
    size_t offset = page_offset + lane_offset;

    it->second.mmap_file.value()->prefetch(offset, per_kv_token_mem_size_);
  }
}

void ZenFileSystem::hintsPurge(vp_addr_t token_lane_addr) {
  // 1. Find which blob we are.
  vp_blob_addr_t blob_addr = getBlobAddr(token_lane_addr, options_.page_bits, options_.lane_bits);

  // 2. If this blob in mmap file, we need to purge.
  auto it = blob_.find(blob_addr);
  if (it != blob_.end() && it->second.type == ZenFSBlobType::kOnDiskMMAP && it->second.mmap_file.has_value()) {
    // Calculate the offset and length for prefetch
    // Each lane represents a token, and we need to prefetch based on the token lane address
    vp_page_addr_t page_addr = getPageAddr(token_lane_addr, options_.page_bits, options_.lane_bits);
    vp_lane_addr_t lane_addr = getLaneAddr(token_lane_addr, options_.page_bits, options_.lane_bits);

    // Calculate offset: page offset + lane offset
    size_t page_offset = page_addr * per_kv_token_mem_size_;
    size_t lane_offset = lane_addr * per_kv_token_mem_size_;
    size_t offset = page_offset + lane_offset;

    it->second.mmap_file.value()->purge(offset, per_kv_token_mem_size_);
  }
}

vp_addr_t ZenFileSystem::malloc() {
  for (auto& [f, s] : blob_) {
    vp_addr_t ret = -1;
    _findFreeAddrInBlob(f, &ret);
    if (ret != -1) { return ret; }
  }

  // 2. Create a new blob
  if (blob_.size() >= 1) {
    switch (options_.mmap_type) {
      case ZenFSBlobMMapType::kFile: {
        _createBlobOnDisk();
        break;
      }
      case ZenFSBlobMMapType::kAnonymous: {
        _createBlobOnAnonymousFile();
        break;
      }
    }
  }

  // 3. Find again !!!
  for (auto& [f, s] : blob_) {
    vp_addr_t ret = -1;
    _findFreeAddrInBlob(f, &ret);
    if (ret != -1) { return ret; }
  }

  MLLM_ERROR_EXIT(ExitCode::kCoreError, "No more blob or some inner error in ZenFS");
  return -1;
}

void ZenFileSystem::free(vp_addr_t addr) {
  // Extract blob address from the virtual address
  vp_blob_addr_t blob_addr = getBlobAddr(addr, options_.page_bits, options_.lane_bits);

  // Get the lane address to determine which bit to free in the bitmap
  vp_lane_addr_t lane_addr = getLaneAddr(addr, options_.page_bits, options_.lane_bits);

  // Find the blob
  auto it = blob_.find(blob_addr);
  if (it != blob_.end()) {
    // Calculate which word in the bitmap and which bit within that word
    size_t word_index = lane_addr / 64;
    size_t bit_index = lane_addr % 64;

    // Set the bit to 1 to indicate it's free
    if (word_index < it->second.free_lanes_bits_map.size()) {
      it->second.free_lanes_bits_map[word_index] |= (1ULL << bit_index);
    }
  }
}

char* ZenFileSystem::access(vp_addr_t addr) {
  // Extract components from the virtual address
  vp_blob_addr_t blob_addr = getBlobAddr(addr, options_.page_bits, options_.lane_bits);
  vp_page_addr_t page_addr = getPageAddr(addr, options_.page_bits, options_.lane_bits);
  vp_lane_addr_t lane_addr = getLaneAddr(addr, options_.page_bits, options_.lane_bits);

  // Find the blob
  auto it = blob_.find(blob_addr);
  if (it != blob_.end()) {
    // Get the tensor data pointer
    char* data_ptr = it->second.data.ptr<char>();
    if (data_ptr) {
      // Calculate the offset to the specific lane
      // Each lane represents a token, and each token has per_kv_token_mem_size_ bytes
      size_t offset = (page_addr * (1 << options_.lane_bits) + lane_addr) * per_kv_token_mem_size_;
      return data_ptr + offset;
    }
  }

  return nullptr;
}

void ZenFileSystem::_createBlobOnDisk() {
  // Calculate blob size.
  size_t total_bits = (1 << (options_.page_bits + options_.lane_bits));
  size_t uint64_count = (total_bits + sizeof(uint64_t) - 1) / sizeof(uint64_t);

  // Blob size is not K and V.
  // Is 1 << (options_.page_bits + options_.lane_bits) * elements * sizeof(dtype).
  // K and V shared one blob.
  size_t blob_size = (1 << (options_.page_bits + options_.lane_bits)) * options_.per_k_token_ele * bytesOfType(options_.k_dtype)
                     / lanesOfType(options_.v_dtype);

  std::error_code ec;
  auto mmap_file = ZenFSBlobMMAPFile::create(blob_size, ZenFSMMAPMode::kReadWrite,
                                             options_.working_dir + "/blob_" + std::to_string(blob_.size()) + ".kv", ec);

  if (ec) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Failed to create mmap file for blob on disk"); }

  // Create Tensor on MMAP
  auto this_tensor_blob_id = blob_.size();
  TensorViewImpl::shape_t shape{(1 << (options_.page_bits + options_.lane_bits)), (int32_t)options_.per_k_token_ele};
  auto s = TensorStorage::create(shape, options_.k_dtype, kCPU);
  auto t = TensorViewImpl::create(shape, s);
  s->name_ = std::to_string(this_tensor_blob_id);
  s->ptr_ = mmap_file->data();
  s->mem_type_ = kParamsMMAP;
  auto tensor = Tensor(t);

  // Create new blob descriptor
  blob_.insert({(uint32_t)this_tensor_blob_id, ZenFSBlob{.data = std::move(tensor),
                                                         .type = ZenFSBlobType::kOnDiskMMAP,
                                                         .mmap_file = mmap_file,
                                                         .free_lanes_bits_map = std::vector<uint64_t>(uint64_count, ~0ULL)}});
}

void ZenFileSystem::_createBlobOnAnonymousFile() {
  // Calculate blob size.
  size_t total_bits = (1 << (options_.page_bits + options_.lane_bits));
  size_t uint64_count = (total_bits + 64 - 1) / 64;

  // Blob size is not K and V.
  // Is 1 << (options_.page_bits + options_.lane_bits) * elements * sizeof(dtype).
  // K and V shared one blob.
  size_t blob_size = (1 << (options_.page_bits + options_.lane_bits)) * options_.per_k_token_ele * bytesOfType(options_.k_dtype)
                     / lanesOfType(options_.k_dtype);

  std::error_code ec;
  auto mmap_file = ZenFSBlobMMAPFile::create(blob_size, ZenFSMMAPMode::kAnonymous, "", ec);

  if (ec) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Failed to create anonymous mmap file for blob"); }

  // Create Tensor on MMAP
  auto this_tensor_blob_id = blob_.size();
  TensorViewImpl::shape_t shape{(1 << (options_.page_bits + options_.lane_bits)), (int32_t)options_.per_k_token_ele};
  auto s = TensorStorage::create(shape, options_.k_dtype, kCPU);
  auto t = TensorViewImpl::create(shape, s);
  s->name_ = std::to_string(this_tensor_blob_id);
  s->ptr_ = mmap_file->data();
  s->mem_type_ = kParamsMMAP;
  auto tensor = Tensor(t);

  // Create new blob descriptor
  blob_.insert({(uint32_t)this_tensor_blob_id, ZenFSBlob{.data = std::move(tensor),
                                                         .type = ZenFSBlobType::kOnDiskMMAP,
                                                         .mmap_file = mmap_file,
                                                         .free_lanes_bits_map = std::vector<uint64_t>(uint64_count, ~0ULL)}});
}

void ZenFileSystem::_findFreeAddrInBlob(vp_blob_addr_t blob_addr, vp_addr_t* ret_addr) {
  auto& bitmap = blob_[blob_addr].free_lanes_bits_map;
  for (size_t i = 0; i < bitmap.size(); ++i) {
    uint64_t w = bitmap[i];
    if (w != 0) {
      int bit_index = __builtin_ctzll(w);
      bitmap[i] &= ~(1ULL << bit_index);
      *ret_addr = (blob_addr << (options_.lane_bits + options_.page_bits)) | int32_t(i * 64 + bit_index);
      return;
    }
  }
  *ret_addr = -1;
}

}  // namespace mllm::prefix_cache
