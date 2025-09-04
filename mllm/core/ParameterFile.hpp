// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <cstdint>
#include <cstring>
#include <memory>

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/SymbolTable.hpp"
#include "mllm/core/MappedFile.hpp"

namespace mllm {

enum class ModelFileVersion : int32_t {
  kUserTemporary = 0,
  kV1 = 1,
  kV2 = 2,
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

  [[nodiscard]] ModelFileVersion version() const;

  void setMappedFile(const MappedFile::ptr_t& mapped_file);

  MappedFile::ptr_t getMappedFile();

  void push(const std::string& name, const Tensor& tensor);

#ifdef MLLM_ENABLE_PY_MLLM
  void __py_push(const std::string& name, const Tensor& tensor);
#endif

  Tensor pull(const std::string& name);

  [[nodiscard]] bool has(const std::string& name) const;

  void remove(const std::string& name);

  inline Tensor operator[](const std::string& name) { return pull(name); }

  [[nodiscard]] const_iterator begin() const;

  [[nodiscard]] const_iterator end() const;

  [[nodiscard]] const_iterator cbegin() const;

  [[nodiscard]] const_iterator cend() const;

  inline std::unordered_map<std::string, Tensor>& dict() { return data_._ref_raw_data(); }

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
