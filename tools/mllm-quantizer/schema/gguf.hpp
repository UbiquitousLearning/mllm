// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "quantize_base.hpp"

// Q4_0
struct QuantizeImpl_GGUF_Q4_0 final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  static ptr_t create();
};

// Q8_0
struct QuantizeImpl_GGUF_Q8_0 final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  static ptr_t create();
};

// Q2_K
struct QuantizeImpl_GGUF_Q2_K final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  static ptr_t create();
};

// Q3_K
struct QuantizeImpl_GGUF_Q3_K final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  static ptr_t create();
};

// Q4_K
struct QuantizeImpl_GGUF_Q4_K final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  static ptr_t create();
};

// Q6_K
struct QuantizeImpl_GGUF_Q6_K final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  static ptr_t create();
};

// Q8_K
struct QuantizeImpl_GGUF_Q8_K final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;
  static ptr_t create();
};
