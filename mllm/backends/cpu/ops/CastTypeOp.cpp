// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/cpu/ops/CastTypeOp.hpp"
#include "mllm/backends/cpu/kernels/Kernels.hpp"
#include "mllm/core/DataTypes.hpp"

#define __MLLM_CAST_SWITCH_CASE(name_space, case_type, from_dtype, to_dtype)                                                   \
  case case_type:                                                                                                              \
    name_space::CastAny<from_dtype, to_dtype>::cast(i.ptr<from_dtype>(), o.ptr<to_dtype>(), i.numel(), options_.getThreads()); \
    break;

namespace mllm::cpu {

CPUCastTypeOp::CPUCastTypeOp(const aops::CastTypeOpOptions& options) : aops::CastTypeOp(options) {}

void CPUCastTypeOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  auto& i = inputs[0];
  auto& o = outputs[0];

  auto to_dtype = options_.dtype;

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
  switch (i.dtype()) {
    case kFloat32: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_fp32_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_fp32_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_fp32_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_fp32_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_fp32_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_fp32_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_fp32_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_fp32_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_fp32_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kFloat16: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_fp16_t, mllm_fp32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_fp16_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_fp16_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_fp16_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_fp16_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_fp16_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_fp16_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_fp16_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_fp16_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kInt8: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_int8_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_int8_t, mllm_fp32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_int8_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_int8_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_int8_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_int8_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_int8_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_int8_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_int8_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kInt16: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_int16_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_int16_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_int16_t, mllm_fp32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_int16_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_int16_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_int16_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_int16_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_int16_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_int16_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kInt32: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_int32_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_int32_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_int32_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_int32_t, mllm_fp32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_int32_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_int32_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_int32_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_int32_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_int32_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kInt64: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_int64_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_int64_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_int64_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_int64_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_int64_t, mllm_fp32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_int64_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_int64_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_int64_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_int64_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kUInt8: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_uint8_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_uint8_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_uint8_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_uint8_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_uint8_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_uint8_t, mllm_fp32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_uint8_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_uint8_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_uint8_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kUInt16: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_uint16_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_uint16_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_uint16_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_uint16_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_uint16_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_uint16_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_uint16_t, mllm_fp32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_uint16_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_uint16_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kUInt32: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_uint32_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_uint32_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_uint32_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_uint32_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_uint32_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_uint32_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_uint32_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_uint32_t, mllm_fp32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt64, mllm_uint32_t, mllm_uint64_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    case kUInt64: {
      switch (to_dtype) {
        __MLLM_CAST_SWITCH_CASE(arm, kFloat16, mllm_uint64_t, mllm_fp16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt8, mllm_uint64_t, mllm_int8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt16, mllm_uint64_t, mllm_int16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt32, mllm_uint64_t, mllm_int32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kInt64, mllm_uint64_t, mllm_int64_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt8, mllm_uint64_t, mllm_uint8_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt16, mllm_uint64_t, mllm_uint16_t)
        __MLLM_CAST_SWITCH_CASE(arm, kUInt32, mllm_uint64_t, mllm_uint32_t)
        __MLLM_CAST_SWITCH_CASE(arm, kFloat32, mllm_uint64_t, mllm_fp32_t)
        default: NYI("CastTypeOp not implemented for this data type");
      }
      break;
    }
    default: NYI("CastTypeOp not implemented for this data type");
  }
#endif
}

}  // namespace mllm::cpu

#undef __MLLM_CAST_SWITCH_CASE