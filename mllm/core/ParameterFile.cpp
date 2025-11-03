// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <limits>
#include <vector>
#include <fstream>

#include "mllm/utils/Common.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/schema/ModelFileV1.hpp"
#include "mllm/core/schema/ModelFileV2.hpp"

#include "mllm/core/TensorStorage.hpp"
#include "mllm/core/TensorViewImpl.hpp"

namespace mllm {

ParameterFile::ParameterFile(ModelFileVersion v) : version_(v) {}

ParameterFile::ptr_t ParameterFile::create(ModelFileVersion v) { return std::make_shared<ParameterFile>(v); }

ModelFileVersion ParameterFile::version() const { return version_; }

void ParameterFile::setMappedFile(const MappedFile::ptr_t& mapped_file) { mapped_file_ = mapped_file; }

MappedFile::ptr_t ParameterFile::getMappedFile() { return mapped_file_; }

void ParameterFile::push(const std::string& name, const Tensor& tensor) {
  if (data_.has(name)) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Parameter already exists: {}", name);
  } else {
    data_.reg(name, tensor);
  }
}

#ifdef MLLM_ENABLE_PY_MLLM
void ParameterFile::__py_push(const std::string& name, const Tensor& tensor) {
  if (data_.has(name)) {
    MLLM_ERROR_EXIT(ExitCode::kIOError, "Parameter already exists: {}", name);
  } else {
    // The tensor is a py object. And may released in the feature.
    // We need to recreate a Tensor descriptor in c++ side.
    data_.reg(name, Tensor(tensor.impl()));
  }
}
#endif

Tensor ParameterFile::pull(const std::string& name) {
  if (!data_.has(name)) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Parameter does not exist: {}", name); }
  return data_[name];
}

bool ParameterFile::has(const std::string& name) const { return data_._raw_data().count(name); }

void ParameterFile::remove(const std::string& name) { data_.remove(name); }

ParameterFile::const_iterator ParameterFile::begin() const { return data_.begin(); }

ParameterFile::const_iterator ParameterFile::end() const { return data_.end(); }

ParameterFile::const_iterator ParameterFile::cbegin() const { return data_.begin(); }

ParameterFile::const_iterator ParameterFile::cend() const { return data_.end(); }

//===----------------------------------------------------------------------===//
// CPU ModelFileV1 ParameterFile
//===----------------------------------------------------------------------===//
ParameterFile::ptr_t ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV1>::read(const std::string& file_path,
                                                                                         bool mmap) {
  auto p_file = ParameterFile::create(ModelFileVersion::kV1);

  if (mmap) {
    auto mmap_file = MappedFile::create(file_path);

    // Set mapped file
    p_file->setMappedFile(mmap_file);

    // Process Header
    uint64_t parameter_desc_offset = -1;
    {
      auto* header = static_cast<ModelFileV1Descriptor*>(mmap_file->data());
      MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V1_MAGIC_NUMBER, header->magic_number);
      parameter_desc_offset = header->parameter_desc_offset;
    }
    char* current_pos = static_cast<char*>(mmap_file->data()) + sizeof(ModelFileV1Descriptor);
    char* end_pos = current_pos + parameter_desc_offset;

    // Loop to find each tensor
    std::vector<ModelFileV1ParamsDescriptorHelper> tensor_meta_info;
    while (current_pos < end_pos) {
      ModelFileV1ParamsDescriptorHelper helper;

      // Gen Name length
      helper.descriptor_.name_len = *reinterpret_cast<uint32_t*>(current_pos);
      current_pos += sizeof(uint32_t);

      // Get Name
      helper.name_ = std::string(current_pos, helper.descriptor_.name_len);
      current_pos += helper.descriptor_.name_len;

      // Weight length
      helper.descriptor_.data_len = *reinterpret_cast<uint64_t*>(current_pos);
      current_pos += sizeof(uint64_t);

      // Offset
      helper.descriptor_.offset = *reinterpret_cast<uint64_t*>(current_pos);
      helper.ptr_ = static_cast<char*>(mmap_file->data()) + helper.descriptor_.offset;
      current_pos += sizeof(uint64_t);

      // Datatypes
      helper.descriptor_.dtype = *reinterpret_cast<int32_t*>(current_pos);
      current_pos += sizeof(int32_t);

      // emplace
      tensor_meta_info.emplace_back(helper);
    }

    for (auto& desc : tensor_meta_info) {
      MLLM_RT_ASSERT(std::numeric_limits<int32_t>::max() >= desc.descriptor_.name_len);

      auto dtype = static_cast<DataTypes>(desc.descriptor_.dtype);
      TensorViewImpl::shape_t shape = {(int32_t)(desc.descriptor_.data_len / bytesOfType(dtype) * lanesOfType(dtype))};
      auto s = TensorStorage::create(shape, dtype, kCPU);
      auto t = TensorViewImpl::create(shape, s);

      s->name_ = desc.name_;
      s->ptr_ = desc.ptr_;
      s->mem_type_ = kParamsMMAP;

      p_file->push(desc.name_, Tensor(t));
    }

  } else {
    // Non-mmap implementation
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file for reading: {}", file_path); }

    // Process Header
    ModelFileV1Descriptor header{};
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V1_MAGIC_NUMBER, header.magic_number);

    uint64_t parameter_desc_offset = header.parameter_desc_offset;
    char* buffer = new char[parameter_desc_offset];
    file.read(buffer, parameter_desc_offset);

    char* current_pos = buffer;
    char* end_pos = current_pos + parameter_desc_offset;

    // Loop to find each tensor
    std::vector<ModelFileV1ParamsDescriptorHelper> tensor_meta_info;
    while (current_pos < end_pos) {
      ModelFileV1ParamsDescriptorHelper helper;

      // Gen Name length
      helper.descriptor_.name_len = *reinterpret_cast<uint32_t*>(current_pos);
      current_pos += sizeof(uint32_t);

      // Get Name
      helper.name_ = std::string(current_pos, helper.descriptor_.name_len);
      current_pos += helper.descriptor_.name_len;

      // Weight length
      helper.descriptor_.data_len = *reinterpret_cast<uint64_t*>(current_pos);
      current_pos += sizeof(uint64_t);

      // Offset
      helper.descriptor_.offset = *reinterpret_cast<uint64_t*>(current_pos);
      current_pos += sizeof(uint64_t);

      // Datatypes
      helper.descriptor_.dtype = *reinterpret_cast<int32_t*>(current_pos);
      current_pos += sizeof(int32_t);

      // emplace
      tensor_meta_info.emplace_back(helper);
    }

    delete[] buffer;

    // Read tensor data
    for (auto& desc : tensor_meta_info) {
      MLLM_RT_ASSERT(std::numeric_limits<int32_t>::max() >= desc.descriptor_.name_len);

      auto dtype = static_cast<DataTypes>(desc.descriptor_.dtype);
      TensorViewImpl::shape_t shape = {(int32_t)(desc.descriptor_.data_len / bytesOfType(dtype) * lanesOfType(dtype))};

      // Seek to data position and read
      file.seekg(desc.descriptor_.offset, std::ios::beg);
      auto s = TensorStorage::create(shape, dtype, kCPU);
      auto t = TensorViewImpl::create(shape, s);

      // Allocate memory and read data
      s->name_ = desc.name_;
      s->mem_type_ = kParamsNormal;
      p_file->push(desc.name_, Tensor(t).alloc());
      file.read(static_cast<char*>(s->ptr_), desc.descriptor_.data_len);
    }

    file.close();
  }

  return p_file;
}

void ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV1>::write(const ParameterFile::ptr_t& parameter_file,
                                                                          const std::string& file_path) {
  std::ofstream out_file(file_path, std::ios::binary);
  if (!out_file.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file for writing: {}", file_path); }

  size_t header_size = sizeof(ModelFileV1Descriptor);
  size_t param_desc_total_size = 0;

  // Calculate total size of the descriptor section
  for (const auto& pair : *parameter_file) {
    const auto& tensor = pair.second;
    param_desc_total_size += sizeof(uint32_t);                        // name length
    param_desc_total_size += tensor.impl()->storage()->name_.size();  // name string
    param_desc_total_size += sizeof(uint64_t);                        // data length
    param_desc_total_size += sizeof(uint64_t);                        // offset
    param_desc_total_size += sizeof(int32_t);                         // data type
  }

  // Write header: parameter_desc_offset is now the SIZE of the descriptor section
  ModelFileV1Descriptor header{};
  header.magic_number = MLLM_MODEL_FILE_V1_MAGIC_NUMBER;
  header.parameter_desc_offset = param_desc_total_size;  // FIXED: size of descriptors (without header)
  out_file.write(reinterpret_cast<const char*>(&header), sizeof(header));

  // Write parameter descriptors and calculate data offsets
  uint64_t current_data_offset = header_size + param_desc_total_size;  // Absolute offset for tensor data
  for (const auto& pair : *parameter_file) {
    const auto& name = pair.first;
    const auto& tensor = pair.second;

    // Name length
    uint32_t name_len = name.size();
    out_file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));

    // Name
    out_file.write(name.c_str(), name_len);

    // Data length (bytes)
    uint64_t data_len = tensor.bytes();
    out_file.write(reinterpret_cast<const char*>(&data_len), sizeof(data_len));

    // Offset (absolute from file start)
    out_file.write(reinterpret_cast<const char*>(&current_data_offset), sizeof(current_data_offset));
    current_data_offset += data_len;

    // Data type
    int32_t dtype = static_cast<int32_t>(tensor.dtype());
    out_file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
  }

  // Write tensor data
  for (const auto& pair : *parameter_file) {
    const auto& tensor = pair.second;
    size_t data_size = tensor.bytes();
    out_file.write(reinterpret_cast<const char*>(tensor.ptr<uint8_t>()), data_size);
  }

  out_file.close();
}

//===----------------------------------------------------------------------===//
// CPU ModelFileV2 ParameterFile
//===----------------------------------------------------------------------===//
ParameterFile::ptr_t ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV2>::read(const std::string& file_path,
                                                                                         bool mmap) {
  auto p_file = ParameterFile::create(ModelFileVersion::kV2);

  if (mmap) {
    auto mmap_file = MappedFile::create(file_path);

    // Set mapped file
    p_file->setMappedFile(mmap_file);

    // Process Header
    {
      auto* header = static_cast<ModelFileV2Descriptor*>(mmap_file->data());
      MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V2_MAGIC_NUMBER, header->magic_number);
      MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V2_VERSION, header->version);
    }

    auto* header = static_cast<ModelFileV2Descriptor*>(mmap_file->data());
    char* params_desc_begin = static_cast<char*>(mmap_file->data()) + header->params_desc_offset;

    if (header->params_desc_offset != sizeof(ModelFileV2Descriptor)) {
      MLLM_WARN(
          "File {} has extra data segment, you can write your own ParameterFileIOImpl to handle it. The default behavior is "
          "to ignore it",
          file_path);
    }

    // Loop through each parameter descriptor
    for (uint32_t i = 0; i < header->num_params; i++) {
      auto* param_desc =
          reinterpret_cast<ModelFileV2ParamsDescriptor*>(params_desc_begin + i * sizeof(ModelFileV2ParamsDescriptor));

      // Extract name
      std::string name(param_desc->_param_name_view());

      // Build shape
      MLLM_RT_ASSERT(param_desc->shape_len <= MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH);
      TensorViewImpl::shape_t shape;
      for (size_t j = 0; j < param_desc->shape_len && j < MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH; j++) {
        shape.push_back(param_desc->shape[j]);
      }

      // Create tensor storage and view
      auto s = TensorStorage::create(shape, static_cast<DataTypes>(param_desc->parameter_type), kCPU);
      auto t = TensorViewImpl::create(shape, s);

      s->name_ = name;
      s->ptr_ = static_cast<char*>(mmap_file->data()) + param_desc->parameter_offset;
      s->mem_type_ = kParamsMMAP;

      // Wrap to Tensor
      auto tensor = Tensor(t);

      // Check parameter size is right
      MLLM_RT_ASSERT(param_desc->parameter_size >= tensor.bytes());

      p_file->push(name, tensor);
    }

  } else {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "Failed to open parameter file"); }

    ModelFileV2Descriptor header;
    file.read(reinterpret_cast<char*>(&header), sizeof(ModelFileV2Descriptor));

    MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V2_MAGIC_NUMBER, header.magic_number);
    MLLM_RT_ASSERT_EQ(MLLM_MODEL_FILE_V2_VERSION, header.version);

    if (header.params_desc_offset != sizeof(ModelFileV2Descriptor)) {
      MLLM_WARN(
          "File {} has extra data segment, you can write your own ParameterFileIOImpl to handle it. The default behavior is "
          "to ignore it",
          file_path);
    }

    std::vector<ModelFileV2ParamsDescriptor> param_descriptors(header.num_params);
    file.seekg(header.params_desc_offset);
    file.read(reinterpret_cast<char*>(param_descriptors.data()), header.num_params * sizeof(ModelFileV2ParamsDescriptor));

    for (uint32_t i = 0; i < header.num_params; i++) {
      const auto& param_desc = param_descriptors[i];

      std::string name(param_desc._param_name_view());

      MLLM_RT_ASSERT(param_desc.shape_len <= MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH);
      TensorViewImpl::shape_t shape;
      for (size_t j = 0; j < param_desc.shape_len && j < MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH; j++) {
        shape.push_back(param_desc.shape[j]);
      }

      auto s = TensorStorage::create(shape, static_cast<DataTypes>(param_desc.parameter_type), kCPU);
      auto t = TensorViewImpl::create(shape, s);

      s->name_ = name;
      auto tensor = Tensor(t).alloc();
      file.seekg(param_desc.parameter_offset);
      file.read(static_cast<char*>(s->ptr_), param_desc.parameter_size);
      s->mem_type_ = kParamsNormal;
      MLLM_RT_ASSERT(param_desc.parameter_size >= tensor.bytes());
      p_file->push(name, tensor);
    }

    file.close();
  }
  return p_file;
}

void ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV2>::write(const ParameterFile::ptr_t& parameter_file,
                                                                          const std::string& file_path) {
  std::ofstream out_file(file_path, std::ios::binary);
  if (!out_file.is_open()) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file for writing: {}", file_path); }

  // Calculate header
  ModelFileV2Descriptor header{};
  header.magic_number = MLLM_MODEL_FILE_V2_MAGIC_NUMBER;
  header.version = MLLM_MODEL_FILE_V2_VERSION;
  header.num_params = parameter_file->dict().size();
  header.params_desc_offset = sizeof(ModelFileV2Descriptor);  // No extra data segment

  // Write header
  out_file.write(reinterpret_cast<const char*>(&header), sizeof(header));

  // Write parameter descriptors and tensor data
  size_t current_param_desc_offset = sizeof(ModelFileV2Descriptor);
  size_t current_data_offset =
      sizeof(ModelFileV2Descriptor) + parameter_file->dict().size() * sizeof(ModelFileV2ParamsDescriptor);

  // First pass: write parameter descriptors
  std::vector<std::pair<std::string, Tensor>> tensors;
  for (const auto& pair : *parameter_file) { tensors.emplace_back(pair.first, pair.second); }

  // Write parameter descriptors
  for (const auto& [name, tensor] : tensors) {
    ModelFileV2ParamsDescriptor param_desc{};

    // Set parameter id (using index in the list)
    param_desc.parameter_id = &tensor - &tensors[0].second;  // This is a simple way to get index

    // Set parameter type
    param_desc.parameter_type = static_cast<uint32_t>(tensor.dtype());

    // Set parameter size
    param_desc.parameter_size = tensor.bytes();

    // Set parameter offset
    param_desc.parameter_offset = current_data_offset;

    // Set shape
    const auto& shape = tensor.shape();
    param_desc.shape_len = shape.size();
    for (size_t i = 0; i < shape.size() && i < MLLM_MODEL_FILE_V2_TENSOR_SHAPE_LENGTH; i++) { param_desc.shape[i] = shape[i]; }

    // Set name
    strncpy(param_desc.name, name.c_str(), MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH - 1);
    param_desc.name[MLLM_MODEL_FILE_V2_PARAMS_NAME_LENGTH - 1] = '\0';

    // Write parameter descriptor
    out_file.write(reinterpret_cast<const char*>(&param_desc), sizeof(param_desc));

    // Update current data offset
    current_data_offset += tensor.bytes();
  }

  // Second pass: write tensor data
  for (const auto& [name, tensor] : tensors) {
    size_t data_size = tensor.bytes();
    out_file.write(reinterpret_cast<const char*>(tensor.ptr<uint8_t>()), data_size);
  }

  out_file.close();
}
}  // namespace mllm
