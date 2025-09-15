// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>

#include "mllm/mllm.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ffi {
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

}  // namespace mllm::ffi
