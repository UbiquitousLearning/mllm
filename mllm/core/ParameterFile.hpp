/**
 * @file ParameterFile.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#pragma once

#include <string>
#include <cstdint>
#include <cstring>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/SymbolTable.hpp"

namespace mllm {

enum class ModelFileVersion : int32_t {
  kUserTemporary = 0,
  kV1 = 1,
  kV2 = 2,
};

//===----------------------------------------------------------------------===//
// MappedFile
//===----------------------------------------------------------------------===//
class MappedFile {
 public:
  using ptr_t = std::shared_ptr<MappedFile>;

  explicit MappedFile(const std::string& filename);

  static ptr_t create(const std::string& filename);

  ~MappedFile();

  [[nodiscard]] inline void* data() const { return mapping_; }

  [[nodiscard]] inline size_t size() const { return size_; }

 private:
  int fd_ = -1;
  size_t size_ = 0;
  void* mapping_ = nullptr;
};

//===----------------------------------------------------------------------===//
// ParameterFile
//===----------------------------------------------------------------------===//
class ParameterFile {
 public:
  using ptr_t = std::shared_ptr<ParameterFile>;
  using const_iterator = typename SymbolTable<std::string, Tensor>::const_iterator;

  explicit ParameterFile(ModelFileVersion v = ModelFileVersion::kUserTemporary);

  static ptr_t create(ModelFileVersion v = ModelFileVersion::kUserTemporary);

  ModelFileVersion version() const;

  void setMappedFile(const MappedFile::ptr_t& mapped_file);

  MappedFile::ptr_t getMappedFile();

  void push(const std::string& name, const Tensor& tensor);

  Tensor pull(const std::string& name);

  bool has(const std::string& name) const;

  inline Tensor operator[](const std::string& name) { return pull(name); }

  const_iterator begin() const;

  const_iterator end() const;

  const_iterator cbegin() const;

  const_iterator cend() const;

 private:
  MappedFile::ptr_t mapped_file_ = nullptr;
  ModelFileVersion version_ = ModelFileVersion::kUserTemporary;
  SymbolTable<std::string, Tensor> data_;
};

template<DeviceTypes __device_type, ModelFileVersion __model_file_version>
struct ParameterFileIOImpl {
  static ParameterFile::ptr_t read(const std::string& file_path);

  static void write(const ParameterFile::ptr_t& parameter_file, const std::string& file_path);
};

template<>
struct ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV1> {
  static ParameterFile::ptr_t read(const std::string& file_path);

  static void write(const ParameterFile::ptr_t& parameter_file, const std::string& file_path);
};

template<>
struct ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV2> {
  static ParameterFile::ptr_t read(const std::string& file_path);

  static void write(const ParameterFile::ptr_t& parameter_file, const std::string& file_path);
};

}  // namespace mllm
