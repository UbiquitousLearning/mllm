// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <cstring>
#include <tvm/ffi/reflection/registry.h>

#include "mllm/core/aops/SoftmaxOp.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/ffi/Object.hh"

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::ObjectDef<::mllm::ffi::SoftmaxOpOptionsObj>().def_static("__create__", [](int dim) -> mllm::ffi::SoftmaxOpOptions {
    auto v = ::mllm::aops::SoftmaxOpOptions{.axis = dim};
    return mllm::ffi::SoftmaxOpOptions(v);
  });

  refl::ObjectDef<::mllm::ffi::SoftmaxOpObj>();
  refl::GlobalDef().def(
      "mllm.aops.__ctx_create_softmax_op", [](const mllm::ffi::Device& d, const mllm::ffi::SoftmaxOpOptions& o) {
        auto v = mllm::Context::instance().getBackend(d.get()->device)->createOp(mllm::OpTypes::kSoftmax, o.get()->options_);
        return mllm::ffi::BaseOp(v);
      });
  // ===============================================================
  // Dispatcher things
  // ===============================================================
  refl::GlobalDef().def("mllm.engine.dispatch", [](const mllm::ffi::Device& d, const mllm::ffi::BaseOp& op,
                                                   const tvm::ffi::Array<mllm::ffi::Tensor>& input_ffi) {
    mllm::DispatcherManager::dispatcher_id_t id = (int32_t)(d.get()->device);
    std::vector<mllm::Tensor> inputs;
    for (auto& t : input_ffi) { inputs.push_back(t.get()->mllm_tensor_); }
    auto task = mllm::Task::createExecuteOpTask(op.get()->op_ptr_, inputs, {});
    mllm::Context::instance().dispatcherManager()->submit(id, task);
    tvm::ffi::Array<mllm::ffi::Tensor> ret;
    for (auto& o : task->outputs) { ret.push_back(mllm::ffi::Tensor(o)); }
    return ret;
  });
}
