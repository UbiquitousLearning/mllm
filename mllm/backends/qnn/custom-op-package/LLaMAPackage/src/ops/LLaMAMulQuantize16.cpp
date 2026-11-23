#include <cmath>
#include <cstddef>
#include <cstdint>

#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"

#ifndef REFERENCE_OP
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#endif

BEGIN_PKG_OP_DEFINITION(PKG_LLaMAMulQuantize16);

namespace {

#ifndef REFERENCE_OP
constexpr size_t kInputsPerVector = 32;
constexpr size_t kOutputsPerVector = 64;
constexpr unsigned int kBytesPerVector = 128;
constexpr int kRoundingFractionBits = 15;

inline HVX_Vector splat(float value) {
  return Q6_V_vsplat_R(__builtin_bit_cast(int32_t, value));
}

inline HVX_Vector multiplyAndScale(HVX_Vector gate, HVX_Vector up, HVX_Vector reciprocal, HVX_Vector half,
                                  HVX_Vector low, HVX_Vector high, HVX_Vector radix) {
  const HVX_Vector product = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(gate, up));
  const HVX_Vector scaled = Q6_Vqf32_vmpy_VsfVsf(product, reciprocal);
  const HVX_Vector biased = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(scaled, half));
  const HVX_Vector clipped = Q6_Vsf_vmax_VsfVsf(Q6_Vsf_vmin_VsfVsf(biased, high), low);
  const HVX_Vector fixed = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(clipped, radix));
  return Q6_Vw_vasr_VwR(Q6_Vw_equals_Vsf(fixed), kRoundingFractionBits);
}

bool mulQuantizeHvx(const float* __restrict gate, const float* __restrict up, uint16_t* __restrict codes,
                   size_t count, float inv_scale, bool use_unsigned_offset) {
  if (!gate || !up || !codes || count % kOutputsPerVector != 0) {
    return false;
  }
  const uintptr_t pointer_bits =
      reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) | reinterpret_cast<uintptr_t>(codes);
  if (pointer_bits % kBytesPerVector != 0) { return false; }
  const auto* gate_vectors = reinterpret_cast<const HVX_Vector*>(gate);
  const auto* up_vectors = reinterpret_cast<const HVX_Vector*>(up);
  auto* code_vectors = reinterpret_cast<HVX_Vector*>(codes);
  const HVX_Vector reciprocal = splat(inv_scale);
  const HVX_Vector half = Q6_Vqf32_vadd_VsfVsf(splat(0.5f), Q6_V_vzero());
  const HVX_Vector low = splat(-32768.0f);
  const HVX_Vector high = splat(32767.0f);
  const HVX_Vector radix = splat(static_cast<float>(1 << kRoundingFractionBits));
  const HVX_Vector offset = Q6_V_vsplat_R(use_unsigned_offset ? 0x80008000U : 0U);

  for (size_t i = 0; i < count / kInputsPerVector; i += 2) {
    const HVX_Vector first = multiplyAndScale(gate_vectors[i], up_vectors[i], reciprocal, half, low, high, radix);
    const HVX_Vector second = multiplyAndScale(gate_vectors[i + 1], up_vectors[i + 1], reciprocal, half, low, high, radix);
    const HVX_Vector packed = Q6_Vh_vpack_VwVw_sat(second, first);
    code_vectors[i / 2] = Q6_Vh_vadd_VhVh(packed, offset);
  }
  return true;
}

#endif

bool contiguous(const Tensor& value) {
  constexpr auto disallowed = Tensor::tformat_is_not_flat | Tensor::tformat_is_chunked | Tensor::tformat_is_indirect;
  return (value.get_tensor_format_code() & disallowed) == 0;
}

GraphStatus mulQuantize16(Tensor& output, const Tensor& gate, const Tensor& up, const PlainFloatTensor& scale) {
  if (gate.get_dtype() != DType::Float32 || up.get_dtype() != DType::Float32 || gate.dims() != up.dims()) {
    return GraphStatus::ErrorDimensions;
  }
  const auto out_type = output.get_dtype();
  const bool offset_binary = out_type == DType::QUInt16;
  if ((!offset_binary && out_type != DType::QInt16) || !contiguous(gate) || !contiguous(up) || !contiguous(output)
      || output.interface_offset() != (offset_binary ? 32768 : 0)) {
    return GraphStatus::ErrorFatal;
  }
  output.set_dims(gate);
  const auto [batch, height, width, depth] = gate.dims();
  const std::size_t count = batch * height * width * depth;
  if (count == 0 || gate.total_storage_elements() < count ||
      up.total_storage_elements() < count || output.total_storage_elements() < count) {
    return GraphStatus::ErrorDimensions;
  }
  const float step = scale(0, 0, 0, 0);
  if (!(step > 0.0f) || !std::isfinite(step) || step != output.interface_scale()) {
    return GraphStatus::ErrorFatal;
  }
  const float reciprocal = 1.0f / step;
  if (!std::isfinite(reciprocal)) { return GraphStatus::ErrorFatal; }
  const auto* left = static_cast<const float*>(gate.raw_data_const());
  const auto* right = static_cast<const float*>(up.raw_data_const());
  auto* codes = static_cast<std::uint16_t*>(output.raw_data());
#ifndef REFERENCE_OP
  // The diagnostic measures the aligned HVX path. An incompatible buffer
  // returns an execution error instead of silently changing dispatch.
  return mulQuantizeHvx(left, right, codes, count, reciprocal, offset_binary)
             ? GraphStatus::Success : GraphStatus::ErrorDimensions;
#else
  if (!left || !right || !codes) { return GraphStatus::ErrorFatal; }
  const int bias = offset_binary ? 32768 : 0;
  for (std::size_t i = 0; i < count; ++i) {
    const float product = left[i] * right[i];
    const float scaled = product * reciprocal;
    if (!std::isfinite(scaled)) { return GraphStatus::ErrorFatal; }
    const float biased = scaled + 0.5f;
    const float clamped = biased < -32768.0f ? -32768.0f : (biased > 32767.0f ? 32767.0f : biased);
    codes[i] = static_cast<std::uint16_t>(static_cast<int>(std::floor(clamped)) + bias);
  }
  return GraphStatus::Success;
#endif
}

}  // namespace

DEF_PACKAGE_OP(mulQuantize16, "LLaMAMulQuantize16")
DEF_PACKAGE_PARAM_ORDER("LLaMAMulQuantize16", "scale", true, nullptr)

END_PKG_OP_DEFINITION(PKG_LLaMAMulQuantize16);
