// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/Layer.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/engine/Task.hpp"

namespace mllm::nn {
void LayerImpl::load(const ParameterFile::ptr_t& ploader) { instanced_op_->load(ploader); }

void LayerImpl::to(DeviceTypes device_type) {
  auto& ctx = Context::instance();
  instanced_op_ = nullptr;
  instanced_op_ = ctx.getBackend(device_type)->createOp(op_type_, options_);
  instanced_op_->setName(getAbsoluteName());

  if (device_type != kCPU && ctx.getBackend(device_type)->isWeightOnDevice()) {
    auto temp_params_loader = ParameterFile::create();

    auto params = instanced_op_->getParams();
    for (auto& param : *params) {
      auto t = param.second;
      if (t) { temp_params_loader->push(param.first, t.to(device_type)); }
    }
    instanced_op_->load(temp_params_loader);
  }

  device_type_ = device_type;
}

OpTypes LayerImpl::opType() const { return op_type_; }

BaseOpOptionsBase& LayerImpl::refOptions() { return options_; }

void LayerImpl::__fmt_print(std::stringstream& ss) {
  for (int i = 0; i < getDepth() * 4; i++) { ss << " "; }
  ss << getAbsoluteName() << ", device: " << deviceTypes2Str(device_type_) << "\n";
}

BaseOp::ptr_t LayerImpl::getInstancedOp() { return instanced_op_; }

void LayerImpl::setInstancedOp(const BaseOp::ptr_t& op) { instanced_op_ = op; }

LayerImpl::ptr_t Layer::impl() const { return impl_; }

Layer::Layer(const LayerImpl::ptr_t& impl) : impl_(impl) {}

std::vector<Tensor> Layer::__main(const std::vector<Tensor>& inputs) {
  auto& ctx = Context::instance();
  auto task = Task::createExecuteOpTask(impl_->getInstancedOp(), inputs, {});

  auto this_thread = Context::instance().thisThread();
  if (this_thread->trace_mode) {
    // Submit!
    // At this moment, heart pounding like thunder
    // Tasks racing through kernels, swift as lightning
    // Threads await, fate hanging by a thread
    // Success or failure in this one moment
    task->custom_context_ptr = this_thread->ir_context.get();
    ctx.dispatcherManager()->submit(Dispatcher::trace_dispatcher_id, task);

    // Everything is Ok. Bravo! You did it.
    // Return what we need.
    return task->outputs;
  } else {
    // Submit!
    // At this moment, heart pounding like thunder
    // Tasks racing through kernels, swift as lightning
    // Threads await, fate hanging by a thread
    // Success or failure in this one moment
    ctx.dispatcherManager()->submit(static_cast<int32_t>(impl_->getInstancedOp()->getDevice()), task);

    // Everything is Ok. Bravo! You did it.
    // Return what we need.
    return task->outputs;
  }
}

OpTypes Layer::opType() const { return impl()->opType(); }

BaseOpOptionsBase& Layer::refOptions() { return impl()->refOptions(); }

Layer& Layer::to(DeviceTypes device_type) {
  impl()->to(device_type);
  return *this;
}

void Layer::__fmt_print(std::stringstream& ss) { impl_->__fmt_print(ss); }

}  // namespace mllm::nn
