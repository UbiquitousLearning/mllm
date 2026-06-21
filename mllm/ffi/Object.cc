// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <cstring>

#include "mllm/ffi/Object.hh"

#include "mllm/core/Tensor.hpp"
#include "mllm/core/TensorStorage.hpp"
#include "mllm/core/TensorViewImpl.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"

namespace mllm::ffi {

::mllm::Tensor __from_dlpack(DLManagedTensor* dl_tensor) {
  auto dl_shape_ptr = dl_tensor->dl_tensor.shape;
  auto dl_shape_size = dl_tensor->dl_tensor.ndim;
  auto dl_dtype = dl_tensor->dl_tensor.dtype;
  auto dl_device = dl_tensor->dl_tensor.device;
  auto dl_offset = dl_tensor->dl_tensor.byte_offset;
  auto dl_stride_ptr = dl_tensor->dl_tensor.strides;
  auto dl_data_ptr = dl_tensor->dl_tensor.data;

  // Mapping DLPack device and dtype into mllm's tensor device and dtype.
  DataTypes dtype;
  switch (dl_dtype.code) {
    case kDLFloat:
      switch (dl_dtype.bits) {
        case 16: dtype = kFloat16; break;
        case 32: dtype = kFloat32; break;
        default: MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported float type"); return ::mllm::Tensor::nil();
      }
      break;
    case kDLInt:
      switch (dl_dtype.bits) {
        case 8: dtype = kInt8; break;
        case 16: dtype = kInt16; break;
        case 32: dtype = kInt32; break;
        case 64: dtype = kInt64; break;
        default: MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported int type"); return ::mllm::Tensor::nil();
      }
      break;
    case kDLUInt:
      switch (dl_dtype.bits) {
        case 8: dtype = kUInt8; break;
        case 16: dtype = kUInt16; break;
        case 32: dtype = kUInt32; break;
        case 64: dtype = kUInt64; break;
        default: MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported uint type"); return ::mllm::Tensor::nil();
      }
      break;
    case kDLBfloat:
      if (dl_dtype.bits == 16) {
        dtype = kBFloat16;
      } else {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported bfloat type");
        return ::mllm::Tensor::nil();
      }
      break;
    default: MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported data type"); return ::mllm::Tensor::nil();
  }

  DeviceTypes device;
  switch (dl_device.device_type) {
    case kDLCPU: device = kCPU; break;
    case kDLCUDA: device = kCUDA; break;
    case kDLOpenCL: device = kOpenCL; break;
    case kDLHexagon: device = kQNN; break;
    default: MLLM_ERROR_EXIT(ExitCode::kCoreError, "Unsupported device type"); return ::mllm::Tensor::nil();
  }

  // Using static ptr_t create(int32_t storage_offset, const shape_t& shape, const stride_t& stride, const storage_t& storage);
  // to create a tensor.

  // Convert shape
  std::vector<int32_t> shape(dl_shape_size);
  for (int i = 0; i < dl_shape_size; ++i) { shape[i] = static_cast<int32_t>(dl_shape_ptr[i]); }

  // Convert stride
  std::vector<int32_t> stride(dl_shape_size);
  for (int i = 0; i < dl_shape_size; ++i) { stride[i] = static_cast<int32_t>(dl_stride_ptr[i]); }

  // Create tensor storage
  auto storage = TensorStorage::create(shape, dtype, device);
  auto impl = TensorViewImpl::create(dl_offset, shape, stride, storage);
  auto ret = ::mllm::Tensor(impl).alloc();

  // Copy data
  switch (device) {
    case kCPU: {
      std::memcpy(ret.ptr<char>(), dl_data_ptr, ret.numel() * bytesOfType(dtype) / lanesOfType(dtype));
      break;
    }
    default: {
      NYI("When copy data from dlpack to mllm, only support cpu device now. You can use .cpu() on torch tensor first before "
          "call this function.");
    }
  }

  return ret;
}

}  // namespace mllm::ffi
