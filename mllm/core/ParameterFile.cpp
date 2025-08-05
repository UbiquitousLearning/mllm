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

MappedFile::MappedFile(const std::string& filename) {
  fd_ = open(filename.c_str(), O_RDONLY);
  if (fd_ == -1) { MLLM_ERROR_EXIT(ExitCode::kIOError, "Failed to open file {}.", filename); }

  struct stat sb{};
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

MappedFile::ptr_t MappedFile::create(const std::string& filename) { return std::make_shared<MappedFile>(filename); }

MappedFile::~MappedFile() {
  if (mapping_) munmap(mapping_, size_);
  if (fd_ != -1) close(fd_);
}

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
ParameterFile::ptr_t ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV1>::read(const std::string& file_path) {
  auto p_file = ParameterFile::create(ModelFileVersion::kV1);
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

    TensorViewImpl::shape_t shape = {(int32_t)desc.descriptor_.data_len};

    auto s = TensorStorage::create(shape, static_cast<DataTypes>(desc.descriptor_.dtype), kCPU);
    auto t = TensorViewImpl::create(shape, s);

    s->name_ = desc.name_;
    s->ptr_ = desc.ptr_;
    s->mem_type_ = kParamsMMAP;

    p_file->push(desc.name_, Tensor(t));
  }

  return p_file;
}

void ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV1>::write(const ParameterFile::ptr_t& parameter_file,
                                                                          const std::string& file_path) {
  // TODO
}

//===----------------------------------------------------------------------===//
// CPU ModelFileV2 ParameterFile
//===----------------------------------------------------------------------===//
ParameterFile::ptr_t ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV2>::read(const std::string& file_path) {
  auto p_file = ParameterFile::create(ModelFileVersion::kV2);
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

  return p_file;
}

void ParameterFileIOImpl<DeviceTypes::kCPU, ModelFileVersion::kV2>::write(const ParameterFile::ptr_t& parameter_file,
                                                                          const std::string& file_path) {
  // TODO
}
}  // namespace mllm