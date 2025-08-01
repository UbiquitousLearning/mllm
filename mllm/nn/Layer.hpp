// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <sstream>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/nn/AbstractNnNode.hpp"
#include "mllm/core/ParameterFile.hpp"

#include "mllm/engine/Task.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::nn {

class LayerImpl : public AbstractNnNode {
 public:
  using ptr_t = std::shared_ptr<LayerImpl>;

  template<typename T>
  LayerImpl(OpTypes op_type, const T& option)
      : AbstractNnNode(AbstractNnNodeTypes::kLayer), op_type_(op_type), options_(option) {}

  ParameterFile::ptr_t refParams();

  void load(const ParameterFile::ptr_t& param_file);

  void to(DeviceTypes device_type);

  [[nodiscard]] OpTypes opType() const;

  BaseOpOptionsBase& refOptions();

  void __fmt_print(std::stringstream& ss);

 private:
  OpTypes op_type_;
  BaseOpOptionsBase options_;
  ParameterFile::ptr_t parameter_loader_;
};

class Layer {
 public:
  template<typename T>
  Layer(OpTypes op_type, const T& cargo) {
    impl_ = std::make_shared<LayerImpl>(op_type, cargo);
  }

  [[nodiscard]] LayerImpl::ptr_t impl() const;

  template<typename... Args>
  Tensor operator()(Args&&... args) {
    auto inputs = std::vector<Tensor>{std::forward<decltype(args)>(args)...};

    auto& ctx = Context::instance();
    auto op = ctx.thisThread()->layer_ops_table[impl_->getAbsoluteName()];
    auto task = Task::createExecuteOpTask(op, inputs, {});

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
      return task->outputs[0];
    } else {
      // Submit!
      // At this moment, heart pounding like thunder
      // Tasks racing through kernels, swift as lightning
      // Threads await, fate hanging by a thread
      // Success or failure in this one moment
      ctx.dispatcherManager()->submit(static_cast<int32_t>(op->getDevice()), task);

      // Everything is Ok. Bravo! You did it.
      // Return what we need.
      return task->outputs[0];
    }
  }

  [[nodiscard]] OpTypes opType() const;

  BaseOpOptionsBase& refOptions();

  Layer& to(DeviceTypes device_type);

  void __fmt_print(std::stringstream& ss);

 private:
  LayerImpl::ptr_t impl_;
};

}  // namespace mllm::nn
