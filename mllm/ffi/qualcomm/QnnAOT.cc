// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <tvm/ffi/any.h>
#include <tvm/ffi/string.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/reflection/registry.h>
#include <memory>

#include "mllm/ffi/qualcomm/QnnAOT.hh"

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<mllm::ffi::QnnAOTEnvObj>().def_static("__create__", [](const std::string& path) -> mllm::ffi::QnnAOTEnv {
    if (path.empty()) {
      auto s = std::make_shared<::mllm::qnn::aot::QnnAOTEnv>();
      return ::mllm::ffi::QnnAOTEnv(s);
    } else {
      auto s = std::make_shared<::mllm::qnn::aot::QnnAOTEnv>(path);
      return ::mllm::ffi::QnnAOTEnv(s);
    }
  });

  refl::GlobalDef().def("mllm.qualcomm.QnnAOTEnv.createContext", [](const mllm::ffi::QnnAOTEnv& self, const std::string& name) {
    auto s = self.get()->qnn_aot_env_ptr_->createContext(name);
    return mllm::ffi::QnnDeviceAndContext(s);
  });
}
