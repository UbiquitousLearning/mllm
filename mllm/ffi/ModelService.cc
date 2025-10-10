// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <tvm/ffi/any.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/reflection/registry.h>

#include "mllm/ffi/Object.hh"

// Qwen3
#include "mllm/models/qwen3/modeling_qwen3_service.hpp"

namespace mllm::ffi {

//===----------------------------------------------------------------------===//
// Qwen3-Model
//===----------------------------------------------------------------------===//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def("mllm.service.session.qwen3", [](const std::string& fp) -> Session {
    auto session = std::make_shared<::mllm::models::qwen3::Qwen3Session>();
    session->fromPreTrain(fp);
    return Session(session);
  });
}

}  // namespace mllm::ffi
