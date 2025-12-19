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

//===----------------------------------------------------------------------===//
// MLLM QcomHTPArch Define
//===----------------------------------------------------------------------===//
class QcomHTPArchObj : public tvm::ffi::Object {
 public:
  mllm::qnn::aot::QcomHTPArch htp_arch_;

  explicit QcomHTPArchObj(const mllm::qnn::aot::QcomHTPArch& obj) : htp_arch_(obj) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.qualcomm.QcomHTPArch", QcomHTPArchObj, tvm::ffi::Object);
};

class QcomHTPArch : public tvm::ffi::ObjectRef {
 public:
  explicit QcomHTPArch(mllm::qnn::aot::QcomHTPArch& ptr) { data_ = tvm::ffi::make_object<QcomHTPArchObj>(ptr); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(QcomHTPArch, tvm::ffi::ObjectRef, QcomHTPArchObj);  // NOLINT
};

//===----------------------------------------------------------------------===//
// MLLM QcomChipset Define
//===----------------------------------------------------------------------===//
class QcomChipsetObj : public tvm::ffi::Object {
 public:
  mllm::qnn::aot::QcomChipset chipset_;

  explicit QcomChipsetObj(const mllm::qnn::aot::QcomChipset& obj) : chipset_(obj) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.qualcomm.QcomChipset", QcomChipsetObj, tvm::ffi::Object);
};

class QcomChipset : public tvm::ffi::ObjectRef {
 public:
  explicit QcomChipset(mllm::qnn::aot::QcomChipset& ptr) { data_ = tvm::ffi::make_object<QcomChipsetObj>(ptr); }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(QcomChipset, tvm::ffi::ObjectRef, QcomChipsetObj);  // NOLINT
};

//===----------------------------------------------------------------------===//
// MLLM QcomTryBestPerformance Define
//===----------------------------------------------------------------------===//
class QcomTryBestPerformanceObj : public tvm::ffi::Object {
 public:
  mllm::qnn::aot::QcomTryBestPerformance perf_;

  explicit QcomTryBestPerformanceObj(const mllm::qnn::aot::QcomTryBestPerformance& obj) : perf_(obj) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.qualcomm.QcomTryBestPerformance", QcomTryBestPerformanceObj, tvm::ffi::Object);
};

class QcomTryBestPerformance : public tvm::ffi::ObjectRef {
 public:
  explicit QcomTryBestPerformance(mllm::qnn::aot::QcomTryBestPerformance& ptr) {
    data_ = tvm::ffi::make_object<QcomTryBestPerformanceObj>(ptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(QcomTryBestPerformance, tvm::ffi::ObjectRef, QcomTryBestPerformanceObj);  // NOLINT
};

//===----------------------------------------------------------------------===//
// MLLM QcomSecurityPDSession Define
//===----------------------------------------------------------------------===//
class QcomSecurityPDSessionObj : public tvm::ffi::Object {
 public:
  mllm::qnn::aot::QcomSecurityPDSession pd_;

  explicit QcomSecurityPDSessionObj(const mllm::qnn::aot::QcomSecurityPDSession& obj) : pd_(obj) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.qualcomm.QcomSecurityPDSession", QcomSecurityPDSessionObj, tvm::ffi::Object);
};

class QcomSecurityPDSession : public tvm::ffi::ObjectRef {
 public:
  explicit QcomSecurityPDSession(mllm::qnn::aot::QcomSecurityPDSession& ptr) {
    data_ = tvm::ffi::make_object<QcomSecurityPDSessionObj>(ptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(QcomSecurityPDSession, tvm::ffi::ObjectRef, QcomSecurityPDSessionObj);  // NOLINT
};

//===----------------------------------------------------------------------===//
// MLLM QcomTargetMachine Define
//===----------------------------------------------------------------------===//
class QcomTargetMachineObj : public tvm::ffi::Object {
 public:
  mllm::qnn::aot::QcomTargetMachine target_machine_;

  explicit QcomTargetMachineObj(const mllm::qnn::aot::QcomTargetMachine& obj) : target_machine_(obj) { MLLM_EMPTY_SCOPE; }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.qualcomm.QcomTargetMachine", QcomTargetMachineObj, tvm::ffi::Object);
};

class QcomTargetMachine : public tvm::ffi::ObjectRef {
 public:
  explicit QcomTargetMachine(mllm::qnn::aot::QcomTargetMachine& ptr) {
    data_ = tvm::ffi::make_object<QcomTargetMachineObj>(ptr);
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(QcomTargetMachine, tvm::ffi::ObjectRef, QcomTargetMachineObj);  // NOLINT
};

#endif

}  // namespace mllm::ffi
