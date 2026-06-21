// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <memory>

#include "mllm/mllm.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/DataTypes.hpp"

#include "gguf.hpp"
#include "quantize_base.hpp"

#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q2.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q3.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q4.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q6.hpp"
#include "mllm/backends/cpu/kernels/common/ggml/quantize/quantize_q8.hpp"

namespace {
std::pair<void*, uint64_t> alloc_quant_block(uint64_t count, mllm::DataTypes type) {
  if (count == 0) { return std::make_pair(nullptr, 0); }

  uint64_t lanes = mllm::lanesOfType(type);
  uint64_t bytes_per_block = mllm::bytesOfType(type);

  if (count % lanes != 0) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "Cannot quantize ", std::to_string(count), " elements: not divisible by ",
                    std::to_string(lanes));
  }

  uint64_t num_blocks = count / lanes;
  uint64_t size = num_blocks * bytes_per_block;

  void* data = new char[size];
  return std::make_pair(data, size);
}
}  // namespace

bool QuantizeImpl_GGUF_Q4_0::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "gguf") { return false; }
  if (desc.hints["gguf_type"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "GGUF quantization requires [gguf_type] hint");
  }
  if (desc.hints["gguf_type"] != "Q4_0") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_GGUF_Q4_0::perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  mllm::Tensor weight = mllm::Tensor::nil();

  for (auto& [name, tensor] : params->dict()) {
    if (name.ends_with(".weight")) {
      weight = tensor;
      break;
    }
  }

  if (!weight) {
    MLLM_WARN("No weight found in GGUF Q4_0 quantization");
    return params;
  }

  if (weight.dtype() != MLLM_TYPE_F32) { weight = weight.to(MLLM_TYPE_F32); }

  auto weight_data = weight.ptr<float>();
  uint64_t num_floats = weight.numel();

  auto block = alloc_quant_block(num_floats, MLLM_TYPE_Q4_0);
  void* quant_ptr = block.first;
  uint64_t quant_size = block.second;

  mllm::cpu::quantize_row_q4_0(weight_data, quant_ptr, num_floats);

  mllm::Tensor quantized_weight = mllm::Tensor::empty(weight.shape(), MLLM_TYPE_Q4_0, weight.device()).alloc();
  memcpy(quantized_weight.ptr<void>(), quant_ptr, quant_size);
  quantized_weight.setName(weight.name());

  delete[] (char*)quant_ptr;

  auto ret = mllm::ParameterFile::create();
  ret->push(quantized_weight.name(), quantized_weight);

  return ret;
}

QuantizeImpl::ptr_t QuantizeImpl_GGUF_Q4_0::create() { return std::make_shared<QuantizeImpl_GGUF_Q4_0>(); }

bool QuantizeImpl_GGUF_Q8_0::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "gguf") { return false; }
  if (desc.hints["gguf_type"] != "Q8_0") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_GGUF_Q8_0::perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  mllm::Tensor weight = mllm::Tensor::nil();

  for (auto& [name, tensor] : params->dict()) {
    if (name.ends_with(".weight")) {
      weight = tensor;
      break;
    }
  }

  if (!weight) return params;
  if (weight.dtype() != MLLM_TYPE_F32) weight = weight.to(MLLM_TYPE_F32);

  auto weight_data = weight.ptr<float>();
  uint64_t num_floats = weight.numel();

  auto block = alloc_quant_block(num_floats, MLLM_TYPE_Q8_0);
  void* quant_ptr = block.first;
  uint64_t quant_size = block.second;

  mllm::cpu::quantize_row_q8_0(weight_data, quant_ptr, num_floats);

  mllm::Tensor quantized_weight = mllm::Tensor::empty(weight.shape(), MLLM_TYPE_Q8_0, weight.device()).alloc();
  memcpy(quantized_weight.ptr<void>(), quant_ptr, quant_size);
  quantized_weight.setName(weight.name());

  delete[] (char*)quant_ptr;

  auto ret = mllm::ParameterFile::create();
  ret->push(quantized_weight.name(), quantized_weight);
  return ret;
}

QuantizeImpl::ptr_t QuantizeImpl_GGUF_Q8_0::create() { return std::make_shared<QuantizeImpl_GGUF_Q8_0>(); }

bool QuantizeImpl_GGUF_Q2_K::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "gguf") { return false; }
  if (desc.hints["gguf_type"] != "Q2_K") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_GGUF_Q2_K::perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  mllm::Tensor weight = mllm::Tensor::nil();

  for (auto& [name, tensor] : params->dict()) {
    if (name.ends_with(".weight")) {
      weight = tensor;
      break;
    }
  }

  if (!weight) return params;
  if (weight.dtype() != MLLM_TYPE_F32) weight = weight.to(MLLM_TYPE_F32);

  auto weight_data = weight.ptr<float>();
  uint64_t num_floats = weight.numel();

  auto block = alloc_quant_block(num_floats, MLLM_TYPE_Q2_K);
  void* quant_ptr = block.first;
  uint64_t quant_size = block.second;

  mllm::cpu::quantize_row_q2_K(weight_data, quant_ptr, num_floats);

  mllm::Tensor quantized_weight = mllm::Tensor::empty(weight.shape(), MLLM_TYPE_Q2_K, weight.device()).alloc();
  memcpy(quantized_weight.ptr<void>(), quant_ptr, quant_size);
  quantized_weight.setName(weight.name());

  delete[] (char*)quant_ptr;

  auto ret = mllm::ParameterFile::create();
  ret->push(quantized_weight.name(), quantized_weight);
  return ret;
}

QuantizeImpl::ptr_t QuantizeImpl_GGUF_Q2_K::create() { return std::make_shared<QuantizeImpl_GGUF_Q2_K>(); }

bool QuantizeImpl_GGUF_Q3_K::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "gguf") { return false; }
  if (desc.hints["gguf_type"] != "Q3_K") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_GGUF_Q3_K::perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  mllm::Tensor weight = mllm::Tensor::nil();

  for (auto& [name, tensor] : params->dict()) {
    if (name.ends_with(".weight")) {
      weight = tensor;
      break;
    }
  }

  if (!weight) return params;
  if (weight.dtype() != MLLM_TYPE_F32) weight = weight.to(MLLM_TYPE_F32);

  auto weight_data = weight.ptr<float>();
  uint64_t num_floats = weight.numel();

  auto block = alloc_quant_block(num_floats, MLLM_TYPE_Q3_K);
  void* quant_ptr = block.first;
  uint64_t quant_size = block.second;

  mllm::cpu::quantize_row_q3_K(weight_data, quant_ptr, num_floats);

  mllm::Tensor quantized_weight = mllm::Tensor::empty(weight.shape(), MLLM_TYPE_Q3_K, weight.device()).alloc();
  memcpy(quantized_weight.ptr<void>(), quant_ptr, quant_size);
  quantized_weight.setName(weight.name());

  delete[] (char*)quant_ptr;

  auto ret = mllm::ParameterFile::create();
  ret->push(quantized_weight.name(), quantized_weight);
  return ret;
}

QuantizeImpl::ptr_t QuantizeImpl_GGUF_Q3_K::create() { return std::make_shared<QuantizeImpl_GGUF_Q3_K>(); }

bool QuantizeImpl_GGUF_Q4_K::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "gguf") { return false; }
  if (desc.hints["gguf_type"] != "Q4_K") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_GGUF_Q4_K::perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  mllm::Tensor weight = mllm::Tensor::nil();

  for (auto& [name, tensor] : params->dict()) {
    if (name.ends_with(".weight")) {
      weight = tensor;
      break;
    }
  }

  if (!weight) return params;
  if (weight.dtype() != MLLM_TYPE_F32) weight = weight.to(MLLM_TYPE_F32);

  auto weight_data = weight.ptr<float>();
  uint64_t num_floats = weight.numel();

  auto block = alloc_quant_block(num_floats, MLLM_TYPE_Q4_K);
  void* quant_ptr = block.first;
  uint64_t quant_size = block.second;

  mllm::cpu::quantize_row_q4_K(weight_data, quant_ptr, num_floats);

  mllm::Tensor quantized_weight = mllm::Tensor::empty(weight.shape(), MLLM_TYPE_Q4_K, weight.device()).alloc();
  memcpy(quantized_weight.ptr<void>(), quant_ptr, quant_size);
  quantized_weight.setName(weight.name());

  delete[] (char*)quant_ptr;

  auto ret = mllm::ParameterFile::create();
  ret->push(quantized_weight.name(), quantized_weight);
  return ret;
}

QuantizeImpl::ptr_t QuantizeImpl_GGUF_Q4_K::create() { return std::make_shared<QuantizeImpl_GGUF_Q4_K>(); }

bool QuantizeImpl_GGUF_Q6_K::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "gguf") { return false; }
  if (desc.hints["gguf_type"] != "Q6_K") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_GGUF_Q6_K::perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  mllm::Tensor weight = mllm::Tensor::nil();

  for (auto& [name, tensor] : params->dict()) {
    if (name.ends_with(".weight")) {
      weight = tensor;
      break;
    }
  }

  if (!weight) return params;
  if (weight.dtype() != MLLM_TYPE_F32) weight = weight.to(MLLM_TYPE_F32);

  auto weight_data = weight.ptr<float>();
  uint64_t num_floats = weight.numel();

  auto block = alloc_quant_block(num_floats, MLLM_TYPE_Q6_K);
  void* quant_ptr = block.first;
  uint64_t quant_size = block.second;

  mllm::cpu::quantize_row_q6_K(weight_data, quant_ptr, num_floats);

  mllm::Tensor quantized_weight = mllm::Tensor::empty(weight.shape(), MLLM_TYPE_Q6_K, weight.device()).alloc();
  memcpy(quantized_weight.ptr<void>(), quant_ptr, quant_size);
  quantized_weight.setName(weight.name());

  delete[] (char*)quant_ptr;

  auto ret = mllm::ParameterFile::create();
  ret->push(quantized_weight.name(), quantized_weight);
  return ret;
}

QuantizeImpl::ptr_t QuantizeImpl_GGUF_Q6_K::create() { return std::make_shared<QuantizeImpl_GGUF_Q6_K>(); }

bool QuantizeImpl_GGUF_Q8_K::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "gguf") { return false; }
  if (desc.hints["gguf_type"] != "Q8_K") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_GGUF_Q8_K::perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  mllm::Tensor weight = mllm::Tensor::nil();

  for (auto& [name, tensor] : params->dict()) {
    if (name.ends_with(".weight")) {
      weight = tensor;
      break;
    }
  }

  if (!weight) return params;
  if (weight.dtype() != MLLM_TYPE_F32) weight = weight.to(MLLM_TYPE_F32);

  auto weight_data = weight.ptr<float>();
  uint64_t num_floats = weight.numel();

  auto block = alloc_quant_block(num_floats, MLLM_TYPE_Q8_K);
  void* quant_ptr = block.first;
  uint64_t quant_size = block.second;

  mllm::cpu::quantize_row_q8_K(weight_data, quant_ptr, num_floats);

  mllm::Tensor quantized_weight = mllm::Tensor::empty(weight.shape(), MLLM_TYPE_Q8_K, weight.device()).alloc();
  memcpy(quantized_weight.ptr<void>(), quant_ptr, quant_size);
  quantized_weight.setName(weight.name());

  delete[] (char*)quant_ptr;

  auto ret = mllm::ParameterFile::create();
  ret->push(quantized_weight.name(), quantized_weight);
  return ret;
}

QuantizeImpl::ptr_t QuantizeImpl_GGUF_Q8_K::create() { return std::make_shared<QuantizeImpl_GGUF_Q8_K>(); }
