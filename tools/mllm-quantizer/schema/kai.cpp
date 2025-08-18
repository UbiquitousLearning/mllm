// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <memory>

#include "mllm/mllm.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/core/ParameterFile.hpp"

#include "kai.hpp"
#include "quantize_base.hpp"

bool QuantizeImpl_KAI_fp16_fp16_fp16p_mxk_kxn::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "kai") { return false; }
  if (desc.hints["kai_matmul_triplet"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "KAI quantization requires [kai_matmul_triplet] hint");
  }
  if (desc.hints["kai_matmul_triplet"] != "fp16_fp16_fp16p") { return false; }
  if (desc.hints["kai_matmul_layout"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "KAI quantization requires [kai_matmul_layout] hint");
  }
  if (desc.hints["kai_matmul_layout"] != "mxk_kxn") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_KAI_fp16_fp16_fp16p_mxk_kxn::perform(const QuantizeDescriptor& desc,
                                                                             mllm::ParameterFile::ptr_t params) {
  MLLM_WARN("QuantizeImpl_KAI_fp16_fp16_fp16p_mxk_kxn is not supported, because it has precision error.");
  return params;
}

QuantizeImpl::ptr_t QuantizeImpl_KAI_fp16_fp16_fp16p_mxk_kxn::create() {
  return std::make_shared<QuantizeImpl_KAI_fp16_fp16_fp16p_mxk_kxn>();
}

bool QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_nxk::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "kai") { return false; }
  if (desc.hints["kai_matmul_triplet"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "KAI quantization requires [kai_matmul_triplet] hint");
  }
  if (desc.hints["kai_matmul_triplet"] != "f32_qai8dxp_qsi4c32p") { return false; }
  if (desc.hints["kai_matmul_layout"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "KAI quantization requires [kai_matmul_layout] hint");
  }
  if (desc.hints["kai_matmul_layout"] != "mxk_nxk") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_nxk::perform(const QuantizeDescriptor& desc,
                                                                                  mllm::ParameterFile::ptr_t params) {
  mllm::Tensor weight = mllm::Tensor::nil();
  mllm::Tensor bias = mllm::Tensor::nil();

  for (auto& [name, tensor] : params->dict()) {
    if (name.ends_with(".weight")) {
      weight = tensor;
    } else if (name.ends_with(".bias")) {
      bias = tensor;
    }
  }

  if (weight.dtype() != mllm::kFloat32) { weight = weight.to(mllm::kFloat32); }
  if (bias && (bias.dtype() != mllm::kFloat32)) { bias = bias.to(mllm::kFloat32); }

  // NOTE: You need to reshape tensor if params version is v1
  switch (params->version()) {
    case mllm::ModelFileVersion::kV1: {
      auto shape = desc.hints["shape"];
      MLLM_RT_ASSERT(!shape.is_null());
      weight = weight.view(shape);
      break;
    }
    default: {
      MLLM_EMPTY_SCOPE
      break;
    }
  }

  auto weight_shape = weight.shape();
  auto out_channels = weight_shape[0];
  auto in_channels = weight_shape[1];

  mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles tile_cfg;
  if (desc.hints["kai_matmul_tile_cfg"].is_null()) {
    tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32;
  } else {
    std::string tile_cfg_name = desc.hints["kai_matmul_tile_cfg"];
    if (tile_cfg_name == "qai8dxp1x8_qsi4c32p4x8_1x4x32") {
      tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p4x8_1x4x32;
    } else if (tile_cfg_name == "qai8dxp1x8_qsi4c32p8x8_1x8x32") {
      tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x8_qsi4c32p8x8_1x8x32;
    } else if (tile_cfg_name == "qai8dxp4x8_qsi4c32p4x8_8x4x32") {
      tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_8x4x32;
    } else if (tile_cfg_name == "qai8dxp4x8_qsi4c32p4x8_16x4x32") {
      tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p4x8_16x4x32;
    } else if (tile_cfg_name == "qai8dxp4x8_qsi4c32p8x8_4x8x32") {
      tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp4x8_qsi4c32p8x8_4x8x32;
    } else if (tile_cfg_name == "qai8dxp1x4_qsi4c32p4x4_1x4") {
      tile_cfg = mllm::cpu::arm::KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk::Tiles::qai8dxp1x4_qsi4c32p4x4_1x4;
    }
  }

  // pack_rhs_size return byte size.
  int32_t new_weights_size = kai_helper_.quant_pack_rhs_size(out_channels, in_channels, tile_cfg);

  // NOTE:
  // We used a flatter byte buffer to represent the packed weight.
  // The packed weight can't be read or manipulated as a normal tensor.
  mllm::Tensor new_weights = mllm::Tensor::empty({new_weights_size}, mllm::kByte, mllm::kCPU).alloc();

  // Perform quantize
  kai_helper_.quant_pack_rhs_offline(new_weights.ptr<mllm::mllm_byte_t>(), weight.ptr<mllm::mllm_fp32_t>(),
                                     bias ? bias.ptr<mllm::mllm_fp32_t>() : nullptr, out_channels, in_channels, tile_cfg);

  // Assign new weights to the linear op
  new_weights.setName(weight.name());

  auto ret = mllm::ParameterFile::create();
  ret->push(new_weights.name(), new_weights);
  return ret;
}

QuantizeImpl::ptr_t QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_nxk::create() {
  return std::make_shared<QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_nxk>();
}

bool QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_kxn::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "kai") { return false; }
  if (desc.hints["kai_matmul_triplet"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "KAI quantization requires [kai_matmul_triplet] hint");
  }
  if (desc.hints["kai_matmul_triplet"] != "f32_qai8dxp_qsi4c32p") { return false; }
  if (desc.hints["kai_matmul_layout"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "KAI quantization requires [kai_matmul_layout] hint");
  }
  if (desc.hints["kai_matmul_layout"] != "mxk_kxn") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_kxn::perform(const QuantizeDescriptor& desc,
                                                                                  mllm::ParameterFile::ptr_t params) {
  MLLM_WARN("QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_kxn is not supported, because linear weight layout is mxk @ (nxk)^T. If "
            "you really want to use this methods, pls implement it and send a PR to us. Thanks.");
  return params;
}

QuantizeImpl::ptr_t QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_kxn::create() {
  return std::make_shared<QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_kxn>();
}

bool QuantizeImpl_KAI_f16_qsi8d32p_qai4c32p_mxk_nxk::match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) {
  if (desc.hints["quant_method"] != "kai") { return false; }
  if (desc.hints["kai_matmul_triplet"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "KAI quantization requires [kai_matmul_triplet] hint");
  }
  if (desc.hints["kai_matmul_triplet"] != "f16_qsi8d32p_qai4c32p") { return false; }
  if (desc.hints["kai_matmul_layout"].is_null()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "KAI quantization requires [kai_matmul_layout] hint");
  }
  if (desc.hints["kai_matmul_layout"] != "mxk_nxk") { return false; }
  return true;
}

mllm::ParameterFile::ptr_t QuantizeImpl_KAI_f16_qsi8d32p_qai4c32p_mxk_nxk::perform(const QuantizeDescriptor& desc,
                                                                                   mllm::ParameterFile::ptr_t params) {
  MLLM_WARN("QuantizeImpl_KAI_f16_qsi8d32p_qai4c32p_mxk_nxk is not supported, because it has precision error.");
  return params;
}

QuantizeImpl::ptr_t QuantizeImpl_KAI_f16_qsi8d32p_qai4c32p_mxk_nxk::create() {
  return std::make_shared<QuantizeImpl_KAI_f16_qsi8d32p_qai4c32p_mxk_nxk>();
}
