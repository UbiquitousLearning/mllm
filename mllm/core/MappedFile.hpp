// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <cstring>
#include <memory>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

namespace mllm {

class MappedFile {
 public:
  using ptr_t = std::shared_ptr<MappedFile>;

  explicit MappedFile(const std::string& filename);

  static ptr_t create(const std::string& filename);

  ~MappedFile();

  [[nodiscard]] inline void* data() const { return mapping_; }

  [[nodiscard]] inline size_t size() const { return size_; }

 private:
#ifdef _WIN32
  HANDLE file_handle_ = INVALID_HANDLE_VALUE;
  HANDLE mapping_handle_ = INVALID_HANDLE_VALUE;
#else
  int fd_ = -1;
#endif
  size_t size_ = 0;
  void* mapping_ = nullptr;
};

}  // namespace mllm
