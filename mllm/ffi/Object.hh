// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>

#include "mllm/mllm.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/service/Session.hpp"

namespace mllm::ffi {
//===----------------------------------------------------------------------===//
// MLLM Primitives Define
//===----------------------------------------------------------------------===//
class DeviceObj : public tvm::ffi::Object {
 public:
  DeviceTypes device;

  explicit DeviceObj(::mllm::DeviceTypes device) : device(device) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.Device", DeviceObj, tvm::ffi::Object);
};

class Device : public tvm::ffi::ObjectRef {
 public:
  explicit Device(const ::mllm::DeviceTypes& device_type) { data_ = tvm::ffi::make_object<DeviceObj>(device_type); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Device, tvm::ffi::ObjectRef, DeviceObj);  // NOLINT
};

class DTypeObj : public tvm::ffi::Object {
 public:
  DataTypes dtype;

  explicit DTypeObj(::mllm::DataTypes data_type) : dtype(data_type) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.DType", DTypeObj, tvm::ffi::Object);
};

class DType : public tvm::ffi::ObjectRef {
 public:
  explicit DType(const ::mllm::DataTypes& data_type) { data_ = tvm::ffi::make_object<DTypeObj>(data_type); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(DType, tvm::ffi::ObjectRef, DTypeObj);  // NOLINT
};

//===----------------------------------------------------------------------===//
// MLLM Tensor Define
//===----------------------------------------------------------------------===//
class TensorObj : public tvm::ffi::Object {
 public:
  ::mllm::Tensor mllm_tensor_ = ::mllm::Tensor::nil();

  explicit TensorObj(const ::mllm::Tensor& tensor) : mllm_tensor_(tensor) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.Tensor", TensorObj, tvm::ffi::Object);
};

class Tensor : public tvm::ffi::ObjectRef {
 public:
  explicit Tensor(const ::mllm::Tensor& tensor) { data_ = tvm::ffi::make_object<TensorObj>(tensor); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tensor, tvm::ffi::ObjectRef, TensorObj);  // NOLINT
};

::mllm::Tensor __from_dlpack(DLManagedTensor* dl_tensor);

//===----------------------------------------------------------------------===//
// MLLM Session Define
//===----------------------------------------------------------------------===//
class SessionObj : public tvm::ffi::Object {
 public:
  ::mllm::service::Session::ptr_t session_ptr_ = nullptr;

  explicit SessionObj(const ::mllm::service::Session::ptr_t& session_ptr) : session_ptr_(session_ptr) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.service.Session", SessionObj, tvm::ffi::Object);
};

class Session : public tvm::ffi::ObjectRef {
 public:
  explicit Session(const ::mllm::service::Session::ptr_t& session_ptr) {
    data_ = tvm::ffi::make_object<SessionObj>(session_ptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Session, tvm::ffi::ObjectRef, SessionObj);  // NOLINT
};

}  // namespace mllm::ffi
