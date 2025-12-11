// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>

#ifdef MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#endif

namespace mllm::ffi {

#ifdef MLLM_QUALCOMM_QNN_AOT_ON_X86_ENABLE

//===----------------------------------------------------------------------===//
// MLLM Parameter File Define
//===----------------------------------------------------------------------===//
class QnnAOTEnvObj : public tvm::ffi::Object {
 public:
  ::mllm::qnn::aot::QnnAOTEnv::ptr_t qnn_aot_env_ptr_ = nullptr;

  explicit QnnAOTEnvObj(const ::mllm::qnn::aot::QnnAOTEnv::ptr_t& ptr) : qnn_aot_env_ptr_(ptr) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.qualcomm.QnnAOTEnv", QnnAOTEnvObj, tvm::ffi::Object);
};

class QnnAOTEnv : public tvm::ffi::ObjectRef {
 public:
  explicit QnnAOTEnv(::mllm::qnn::aot::QnnAOTEnv::ptr_t& ptr) { data_ = tvm::ffi::make_object<QnnAOTEnvObj>(ptr); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(QnnAOTEnv, tvm::ffi::ObjectRef, QnnAOTEnvObj);  // NOLINT
};

//===----------------------------------------------------------------------===//
// MLLM QnnDeviceAndContext Define
//===----------------------------------------------------------------------===//
class QnnDeviceAndContextObj : public tvm::ffi::Object {
 public:
  std::shared_ptr<::mllm::qnn::aot::QnnDeviceAndContext> qnn_device_and_context_ptr_ = nullptr;

  explicit QnnDeviceAndContextObj(const std::shared_ptr<::mllm::qnn::aot::QnnDeviceAndContext>& ptr)
      : qnn_device_and_context_ptr_(ptr) {
    MLLM_EMPTY_SCOPE;
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.qualcomm.QnnDeviceAndContext", QnnDeviceAndContextObj, tvm::ffi::Object);
};

class QnnDeviceAndContext : public tvm::ffi::ObjectRef {
 public:
  explicit QnnDeviceAndContext(std::shared_ptr<::mllm::qnn::aot::QnnDeviceAndContext>& ptr) {
    data_ = tvm::ffi::make_object<QnnDeviceAndContextObj>(ptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(QnnDeviceAndContext, tvm::ffi::ObjectRef, QnnDeviceAndContextObj);  // NOLINT
};

#endif

}  // namespace mllm::ffi
