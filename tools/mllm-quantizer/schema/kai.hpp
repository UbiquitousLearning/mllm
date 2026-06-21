// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include "quantize_base.hpp"
#include "mllm/backends/cpu/kernels/arm/linear/kai.hpp"

//===----------------------------------------------------------------------===//
// KaiQuantizationPatterns For Linear Op
//===----------------------------------------------------------------------===//
struct QuantizeImpl_KAI_fp16_fp16_fp16p_mxk_kxn final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;

  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;

  static ptr_t create();

  ::mllm::cpu::arm::KaiLinear_fp16_fp16_fp16p_mxk_kxn kai_helper_;
};

struct QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_nxk final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;

  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;

  static ptr_t create();

  ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk kai_helper_;
};

struct QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_kxn final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;

  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;

  static ptr_t create();

  ::mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_kxn kai_helper_;
};

struct QuantizeImpl_KAI_f16_qsi8d32p_qai4c32p_mxk_nxk final : public QuantizeImpl {
  bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;

  mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) override;

  static ptr_t create();

  ::mllm::cpu::arm::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk kai_helper_;
};
