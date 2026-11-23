// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>
#include <new>

#include "HTP/core/log.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/simple_reg.h"
#include "ShadowActivationMask.hpp"

BEGIN_PKG_OP_DEFINITION(PKG_ShadowCorrectionPacket);

namespace {

using namespace shadow_correction_packet;

bool isFlat(const Tensor& value) {
  constexpr auto incompatible = Tensor::tformat_is_not_flat | Tensor::tformat_is_chunked | Tensor::tformat_is_indirect;
  return (value.get_tensor_format_code() & incompatible) == 0;
}

template<bool use_activation_codes, bool use_activation_mask = false>
GraphStatus packCorrectionImpl(Tensor& output, const Tensor& activation, const Tensor& codes, const Tensor& controls,
                               const Tensor* selection) {
  const auto [batch, height, width, depth] = activation.dims();
  const auto [code_batch, code_height, code_width, code_depth] = codes.dims();
  const auto [control_batch, control_height, control_width, control_depth] = controls.dims();
  const auto [out_batch, out_height, out_spatial_width, out_depth] = output.dims();
  if (batch != 1 || height * width != tile_rows || depth != in_width
      || code_batch != batch || code_height != height || code_width != width || code_depth != out_width
      || out_batch != 1 || out_height != tile_rows || out_spatial_width != 1 || out_depth != words_per_row
      || control_batch * control_height * control_width * control_depth != 3) {
    errlog("ShadowCorrectionPacket geometry activation=%zu:%zu:%zu:%zu codes=%zu:%zu:%zu:%zu controls=%zu:%zu:%zu:%zu output=%zu:%zu:%zu:%zu",
           batch, height, width, depth, code_batch, code_height, code_width, code_depth,
           control_batch, control_height, control_width, control_depth, out_batch, out_height, out_spatial_width, out_depth);
    return GraphStatus::ErrorDimensions;
  }
  const auto code_type = codes.get_dtype();
  const bool offset_binary = code_type == DType::QUInt16;
  if (activation.get_dtype() != DType::Float32 || (!offset_binary && code_type != DType::QInt16)
      || codes.interface_offset() != (offset_binary ? 32768 : 0)
      || controls.get_dtype() != DType::Int32 || output.get_dtype() != DType::Int32
      || !isFlat(output) || !isFlat(activation) || !isFlat(codes) || !isFlat(controls)) {
    errlog("ShadowCorrectionPacket tensor types=%u:%u:%u:%u formats=%x:%x:%x:%x code_offset=%d",
           static_cast<unsigned>(activation.get_dtype()), static_cast<unsigned>(codes.get_dtype()),
           static_cast<unsigned>(controls.get_dtype()), static_cast<unsigned>(output.get_dtype()),
           activation.get_tensor_format_code(), codes.get_tensor_format_code(), controls.get_tensor_format_code(),
           output.get_tensor_format_code(), codes.interface_offset());
    return GraphStatus::ErrorFatal;
  }

  if (output.total_storage_elements() < packet_words || activation.total_storage_elements() < tile_rows * in_width
      || codes.total_storage_elements() < tile_rows * out_width || controls.total_storage_elements() < 3) {
    errlog("ShadowCorrectionPacket storage elements activation=%zu codes=%zu controls=%zu output=%zu",
           activation.total_storage_elements(), codes.total_storage_elements(), controls.total_storage_elements(),
           output.total_storage_elements());
    return GraphStatus::ErrorDimensions;
  }
  auto* destination = output.raw_data();
  const auto* rows = static_cast<const float*>(activation.raw_data_const());
  const auto* projected = static_cast<const std::int16_t*>(codes.raw_data_const());
  const auto* limits = static_cast<const std::uint32_t*>(controls.raw_data_const());
  if (!destination || !rows || !projected || !limits) {
    errlog("ShadowCorrectionPacket missing data activation=%d codes=%d controls=%d output=%d",
           rows == nullptr, projected == nullptr, limits == nullptr, destination == nullptr);
    return GraphStatus::ErrorFatal;
  }

  const uint16_t* activation_codes = nullptr;
  const uint32_t* activation_mask = nullptr;
  if constexpr (use_activation_codes) {
    const Tensor* quantized_activation = selection;
    const auto [q_batch, q_height, q_width, q_depth] = quantized_activation->dims();
    if (q_batch != batch || q_height != height || q_width != width || q_depth != depth
        || quantized_activation->total_storage_elements() < tile_rows * in_width) {
      errlog("ShadowCorrectionPacketGuarded: activation code shape differs from FP32 input");
      return GraphStatus::ErrorDimensions;
    }
    if (quantized_activation->get_dtype() != DType::QUInt16 || quantized_activation->interface_offset() != 32768
        || !isFlat(*quantized_activation) || !quantized_activation->raw_data_const()) {
      errlog("ShadowCorrectionPacketGuarded: expected flat offset-binary INT16 activation codes");
      return GraphStatus::ErrorFatal;
    }
    activation_codes = static_cast<const uint16_t*>(quantized_activation->raw_data_const());
  }
  if constexpr (use_activation_mask) {
    const auto [mask_batch, mask_height, mask_width, mask_depth] = selection->dims();
    if (mask_batch != batch || mask_height != height || mask_width != width || mask_depth != activation_mask_words_per_row
        || selection->total_storage_elements() < tile_rows * activation_mask_words_per_row) {
      return GraphStatus::ErrorDimensions;
    }
    if (selection->get_dtype() != DType::Int32 || !isFlat(*selection) || !selection->raw_data_const()) {
      return GraphStatus::ErrorFatal;
    }
    activation_mask = static_cast<const uint32_t*>(selection->raw_data_const());
  }

  auto* packet = new (destination) correction_packet;
  const auto encoding = offset_binary ? code_encoding::offset_binary : code_encoding::signed_fixed;
  pack_shadow_correction_tile<use_activation_codes, use_activation_mask>(
      rows, projected, limits[0], limits[1], limits[2], packet, encoding, activation_codes, activation_mask);
  return GraphStatus::Success;
}

GraphStatus packCorrection(Tensor& output, const Tensor& activation, const Tensor& codes, const Tensor& controls) {
  return packCorrectionImpl<false>(output, activation, codes, controls, nullptr);
}

GraphStatus packCorrectionGuarded(Tensor& output, const Tensor& activation, const Tensor& codes,
                                 const Tensor& quantized_activation, const Tensor& controls) {
  return packCorrectionImpl<true>(output, activation, codes, controls, &quantized_activation);
}

GraphStatus packCorrectionMasked(Tensor& output, const Tensor& activation, const Tensor& codes,
                                const Tensor& mask, const Tensor& controls) {
  return packCorrectionImpl<false, true>(output, activation, codes, controls, &mask);
}

GraphStatus multiplyAndMask(Tensor& product, Tensor& mask, const Tensor& gate, const Tensor& up, const Tensor& controls) {
  const auto [batch, height, width, depth] = gate.dims();
  const auto [mask_batch, mask_height, mask_width, mask_depth] = mask.dims();
  const auto [control_batch, control_height, control_width, control_depth] = controls.dims();
  if (batch != 1 || height * width != tile_rows || depth != in_width || up.dims() != gate.dims()
      || product.dims() != gate.dims()
      || mask_batch != batch || mask_height != height || mask_width != width || mask_depth != activation_mask_words_per_row
      || control_batch * control_height * control_width * control_depth != 3
      || gate.total_storage_elements() < tile_rows * in_width || up.total_storage_elements() < tile_rows * in_width
      || product.total_storage_elements() < tile_rows * in_width || mask.total_storage_elements() < tile_rows * activation_mask_words_per_row
      || controls.total_storage_elements() < 3) {
    return GraphStatus::ErrorDimensions;
  }
  const auto* left = static_cast<const float*>(gate.raw_data_const());
  const auto* right = static_cast<const float*>(up.raw_data_const());
  auto* values = static_cast<float*>(product.raw_data());
  auto* bitmap = static_cast<uint32_t*>(mask.raw_data());
  const auto* limits = static_cast<const uint32_t*>(controls.raw_data_const());
  if (gate.get_dtype() != DType::Float32 || up.get_dtype() != DType::Float32 || product.get_dtype() != DType::Float32
      || mask.get_dtype() != DType::Int32 || controls.get_dtype() != DType::Int32
      || !isFlat(gate) || !isFlat(up) || !isFlat(product) || !isFlat(mask) || !isFlat(controls)
      || !left || !right || !values || !bitmap || !limits || !valid_controls(limits[0], limits[1], limits[2])) {
    return GraphStatus::ErrorFatal;
  }
  const uint32_t threshold = limits[0] < limits[1] ? limits[0] : limits[1];
  multiply_with_activation_mask(left, right, values, bitmap, threshold);
  return GraphStatus::Success;
}

}  // namespace

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(packCorrection, "ShadowCorrectionPacket", SNAIL, Flags::RESOURCE_HVX)
// The full M256 activation exceeds VTCM. The HVX kernel streams contiguous rows from DDR.
DEF_TENSOR_PROPERTIES(Op("ShadowCorrectionPacket", "activation", "codes", "controls"),
                      Flat("*", "activation", "codes", "controls"),
                      MainMemory("*", "activation", "codes", "controls"))
DEF_PACKAGE_PARAM_ORDER("ShadowCorrectionPacket", "controls", true, nullptr)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(packCorrectionGuarded, "ShadowCorrectionPacketGuarded", SNAIL, Flags::RESOURCE_HVX)
DEF_TENSOR_PROPERTIES(Op("ShadowCorrectionPacketGuarded", "activation", "codes", "quantized_activation", "controls"),
                      Flat("*", "activation", "codes", "quantized_activation", "controls"),
                      MainMemory("*", "activation", "codes", "quantized_activation", "controls"))
DEF_PACKAGE_PARAM_ORDER("ShadowCorrectionPacketGuarded", "controls", true, nullptr)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(packCorrectionMasked, "ShadowCorrectionPacketMasked", SNAIL, Flags::RESOURCE_HVX)
DEF_TENSOR_PROPERTIES(Op("ShadowCorrectionPacketMasked", "activation", "codes", "mask", "controls"),
                      Flat("*", "activation", "codes", "mask", "controls"),
                      MainMemory("*", "activation", "codes", "mask", "controls"))
DEF_PACKAGE_PARAM_ORDER("ShadowCorrectionPacketMasked", "controls", true, nullptr)

DEF_PACKAGE_OP_AND_COST_AND_FLAGS(multiplyAndMask, "ShadowMulActivationMask", SNAIL, Flags::RESOURCE_HVX)
DEF_TENSOR_PROPERTIES(Op("ShadowMulActivationMask", "gate", "up", "controls"), Outputs("product", "mask"),
                      Flat("product", "mask", "gate", "up", "controls"),
                      MainMemory("product", "mask", "gate", "up", "controls"))
DEF_PACKAGE_PARAM_ORDER("ShadowMulActivationMask", "controls", true, nullptr)

END_PKG_OP_DEFINITION(PKG_ShadowCorrectionPacket);
