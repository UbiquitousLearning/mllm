// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <tvm/ffi/object.h>
#include <tvm/ffi/memory.h>

namespace mllm::ffi {
//===----------------------------------------------------------------------===//
// MLLM Tensor Define
//===----------------------------------------------------------------------===//
class TensorObj : public tvm::ffi::Object {
 public:
  TensorObj() = default;
  // TODO

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("mllm.Tensor", TensorObj, tvm::ffi::Object);
};

class Tensor : public tvm::ffi::ObjectRef {
 public:
  // TODO

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(Tensor, tvm::ffi::ObjectRef, TensorObj);  // NOLINT
};

}  // namespace mllm::ffi
