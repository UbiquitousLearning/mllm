// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class LinearImplTypes {
  kLinearImplTypes_Start = 0,
  kDefault,

  // BLAS
  kBLAS,

  kGGUF,

  kMllmBlas,

  kKleidiai_Start,
  kKaiLinear_fp16_fp16_fp16p_mxk_kxn,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32,
  kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4,
  KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4,
  KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4,
  KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm,
  kKleidiai_End,

  kGGUF_Start,
  // Add GGUF quantized linear
  kGGUF_End,

  kQNN_Start,
  kQNN_tensor_symm_w8a16,
  kQNN_tensor_symm_w8a8,
  kQNN_End,

  kLinearImplTypes_End,
};

inline LinearImplTypes str2LinearImplTypes(const std::string& str) {
  static const std::unordered_map<std::string, LinearImplTypes> map = {
      {"Default", LinearImplTypes::kDefault},
      {"BLAS", LinearImplTypes::kBLAS},
      {"GGUF", LinearImplTypes::kGGUF},
      {"MllmBlas", LinearImplTypes::kMllmBlas},
      {"KaiLinear_fp16_fp16_fp16p_mxk_kxn", LinearImplTypes::kKaiLinear_fp16_fp16_fp16p_mxk_kxn},
      {"KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32",
       LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32},
      {"KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32",
       LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32},
      {"KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32",
       LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32},
      {"KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32",
       LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32},
      {"KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32",
       LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32},
      {"KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4",
       LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4},
      {"KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4",
       LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4},
      {"KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4",
       LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4},
      {"KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm",
       LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm},
      {"QNN Linear per-tensor symmectrical W8A16", LinearImplTypes::kQNN_tensor_symm_w8a16},
      {"QNN Linear per-tensor symmectrical W8A8", LinearImplTypes::kQNN_tensor_symm_w8a8}};

  auto it = map.find(str);
  if (it != map.end()) { return it->second; }

  // Return default if not found
  return LinearImplTypes::kDefault;
}

inline std::string LinearImplTypes2Str(LinearImplTypes type) {
  static const std::unordered_map<LinearImplTypes, std::string> map = {
      {LinearImplTypes::kDefault, "Default"},
      {LinearImplTypes::kBLAS, "BLAS"},
      {LinearImplTypes::kGGUF, "GGUF"},
      {LinearImplTypes::kMllmBlas, "MllmBlas"},
      {LinearImplTypes::kKaiLinear_fp16_fp16_fp16p_mxk_kxn, "KaiLinear_fp16_fp16_fp16p_mxk_kxn"},
      {LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32,
       "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p4x8_1x4x32"},
      {LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32,
       "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x8_qsi4c32p8x8_1x8x32"},
      {LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32,
       "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_8x4x32"},
      {LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32,
       "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p4x8_16x4x32"},
      {LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32,
       "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp4x8_qsi4c32p8x8_4x8x32"},
      {LinearImplTypes::kKaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4,
       "KaiLinear_f32_qai8dxp_qsi4c32p_mxk_nxk_qai8dxp1x4_qsi4c32p4x4_1x4"},
      {LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4,
       "KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p1x8_qai4c32p4x8_1x4"},
      {LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4,
       "KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x4_qai4c32p4x4_8x4"},
      {LinearImplTypes::KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm,
       "KaiLinear_f16_qsi8d32p_qai4c32p_mxk_nxk_qsi8d32p4x8_qai4c32p4x8_8x4_i8mm"},
      {LinearImplTypes::kQNN_tensor_symm_w8a16, "QNN Linear per-tensor symmectrical W8A16"},
      {LinearImplTypes::kQNN_tensor_symm_w8a8, "QNN Linear per-tensor symmectrical W8A8"}};

  auto it = map.find(type);
  if (it != map.end()) return it->second;
  return "Default";
}

struct LinearOpOptions : public BaseOpOptions<LinearOpOptions> {
  int32_t in_channels;
  int32_t out_channels;
  bool bias;
  LinearImplTypes impl_type;

  LinearImplTypes qnn_impl_type = LinearImplTypes::kQNN_tensor_symm_w8a16;  // specify Linear type when using QNN
};

class LinearOp : public BaseOp {
 public:
  explicit LinearOp(const LinearOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline const LinearOpOptions& options() const { return options_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  LinearOpOptions options_;
};

}  // namespace mllm::aops
