// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <tvm/ffi/any.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/reflection/registry.h>
#include <fmt/core.h>

#include "mllm/ffi/Object.hh"

namespace mllm::ffi {
//===----------------------------------------------------------------------===//
// Helper Functions. [Test if FFI works]s
//===----------------------------------------------------------------------===//
void echo(const tvm::ffi::String& a) { fmt::print("{}", a.c_str()); }

//===----------------------------------------------------------------------===//
// Tensor FFI Binding
//===----------------------------------------------------------------------===//

}  // namespace mllm::ffi

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("mllm.echo", mllm::ffi::echo);
}
