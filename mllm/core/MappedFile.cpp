// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/core/MappedFile.hpp"

namespace mllm {

#ifdef _WIN32
MappedFile::MappedFile(const std::string& filename) {
  file_handle_ =
      CreateFileA(filename.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file_handle_ == INVALID_HANDLE_VALUE) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {}.", filename); }

  LARGE_INTEGER file_size;
  if (!GetFileSizeEx(file_handle_, &file_size)) {
    CloseHandle(file_handle_);
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to get file size for {}.", filename);
  }
  size_ = static_cast<size_t>(file_size.QuadPart);

  mapping_handle_ = CreateFileMapping(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping_handle_ == nullptr) {
    CloseHandle(file_handle_);
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to create file mapping for {}.", filename);
  }

  mapping_ = MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, size_);
  if (mapping_ == nullptr) {
    CloseHandle(mapping_handle_);
    CloseHandle(file_handle_);
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to map view of file {}.", filename);
  }
}
#else
MappedFile::MappedFile(const std::string& filename) {
  fd_ = open(filename.c_str(), O_RDONLY);
  if (fd_ == -1) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {}.", filename); }

  struct stat sb {};
  if (fstat(fd_, &sb) == -1) {
    close(fd_);
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed stat when open file {}, this file may broken.", filename);
  }
  size_ = sb.st_size;

  mapping_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (mapping_ == MAP_FAILED) {
    close(fd_);
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to map file {} to memory space.", filename);
  }
}
#endif

MappedFile::ptr_t MappedFile::create(const std::string& filename) { return std::make_shared<MappedFile>(filename); }

MappedFile::~MappedFile() {
#ifdef _WIN32
  if (mapping_) UnmapViewOfFile(mapping_);
  if (mapping_handle_ != INVALID_HANDLE_VALUE) CloseHandle(mapping_handle_);
  if (file_handle_ != INVALID_HANDLE_VALUE) CloseHandle(file_handle_);
#else
  if (mapping_) munmap(mapping_, size_);
  if (fd_ != -1) close(fd_);
#endif
}

}  // namespace mllm
