// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/Module.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/cf/Op.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/core/aops/GraphOps.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/nn/AbstractNnNode.hpp"

namespace mllm::nn {
ModuleImpl::ModuleImpl() : AbstractNnNode(AbstractNnNodeTypes::kModule) {}

void ModuleImpl::load(const ParameterFile::ptr_t& param_file) {
  auto& h = refChildNodes();
  for (auto& hb : h) {
    switch (hb->getType()) {
      case AbstractNnNodeTypes::kModule: std::static_pointer_cast<ModuleImpl>(hb)->load(param_file); break;
      case AbstractNnNodeTypes::kLayer: std::static_pointer_cast<LayerImpl>(hb)->load(param_file); break;
    }
  }
}

ParameterFile::ptr_t ModuleImpl::params(ModelFileVersion v) {
  ParameterFile::ptr_t param_file = ParameterFile::create();
  auto& h = refChildNodes();
  for (auto& hb : h) {
    switch (hb->getType()) {
      case AbstractNnNodeTypes::kModule: {
        auto p = std::static_pointer_cast<ModuleImpl>(hb)->params(v);
        for (auto& p : *p) { param_file->push(p.first, p.second); }
        break;
      }
      case AbstractNnNodeTypes::kLayer:
        auto p = std::static_pointer_cast<LayerImpl>(hb)->getInstancedOp()->getParams();
        for (auto& p : *p) { param_file->push(p.first, p.second); }
        break;
    }
  }
  return param_file;
}

void ModuleImpl::to(DeviceTypes device_type) {
  auto& h = refChildNodes();
  for (auto& hb : h) {
    switch (hb->getType()) {
      case AbstractNnNodeTypes::kModule: std::static_pointer_cast<ModuleImpl>(hb)->to(device_type); break;
      case AbstractNnNodeTypes::kLayer: std::static_pointer_cast<LayerImpl>(hb)->to(device_type); break;
    }
  }
  device_type_ = device_type;
}

void ModuleImpl::__fmt_print(std::stringstream& ss) {
  for (int i = 0; i < getDepth() * 4; i++) { ss << " "; }
  ss << "Module: " << getAbsoluteName() << ", device: " << deviceTypes2Str(getDevice()) << "\n";
  for (auto& hb : refChildNodes()) {
    switch (hb->getType()) {
      case AbstractNnNodeTypes::kModule: std::static_pointer_cast<ModuleImpl>(hb)->__fmt_print(ss); break;
      case AbstractNnNodeTypes::kLayer: std::static_pointer_cast<LayerImpl>(hb)->__fmt_print(ss); break;
    }
    ss << "\n";
  }
}

Module::Module(const std::string& name) {
  impl_ = std::make_shared<ModuleImpl>();
  impl()->setName(name);
  impl()->setAbsoluteName(name);
}

ModuleImpl::ptr_t Module::impl() const { return impl_; }

void Module::to(DeviceTypes device_type) { impl()->to(device_type); }

void Module::load(const ParameterFile::ptr_t& param_file) { impl_->load(param_file); }

void Module::__fmt_print(std::stringstream& ss) const { impl()->__fmt_print(ss); }

std::vector<Tensor> Module::__main(const std::vector<Tensor>& inputs) {
  auto& ctx = Context::instance();

  // When enter in Module. Submit a GraphBegin Op
  (void)ctx.buildOpAndSubmitTask(OpTypes::kGraphBegin, aops::GraphBeginOpOptions{.graph_name = impl_->getAbsoluteName()},
                                 inputs);

  auto outs = forward(inputs);

  // When exit from Module. Submit a GraphEnd Op
  (void)ctx.buildOpAndSubmitTask(OpTypes::kGraphEnd, aops::GraphEndOpOptions{.graph_name = impl_->getAbsoluteName()}, inputs);

  return outs;
}

std::vector<Tensor> Module::__trace(const std::vector<Tensor>& inputs) {
  auto ir_ctx = Context::instance().thisThread()->ir_context;

  // Create call graph.
  auto call_op = ir_ctx->create<ir::graph::CallGraphOp>(ir_ctx->create<ir::SymbolAttr>(impl_->getAbsoluteName()));

  // Create subgraph under ModuleOp
  ir::graph::SubGraphOp::ptr_t this_graph_ir = nullptr;
  {
    auto guard = ir::IRWriterGuard(ir_ctx, ir_ctx->topLevelOp()->cast_<ir::ModuleOp>()->getTopRegion());
    this_graph_ir = ir_ctx->create<ir::graph::SubGraphOp>(ir_ctx->create<ir::SymbolAttr>(impl_->getAbsoluteName()), impl_);
  }
  this_graph_ir->setDevice(impl_->getDevice());

  // Wrap the inputs to tensor ir.
  std::vector<ir::tensor::TensorValue::ptr_t> inputs_ir = ir::tensor::wrapTensors2TensorIR(ir_ctx.get(), inputs);

  // Link inputs to subgraph.
  for (const auto& input_ir : inputs_ir) {
    (*input_ir)-- > this_graph_ir;
    (*input_ir)-- > call_op;
    this_graph_ir->getTopRegion()->inputs().push_back(input_ir);
  }

  // Forward
  std::vector<Tensor> outputs;
  {
    ir_ctx->setDevice(impl_->getDevice());
    auto guard = ir::IRWriterGuard(ir_ctx, this_graph_ir->getTopRegion());
    outputs = forward(inputs);
  }

  // wrap the outputs to tensor ir.
  std::vector<std::shared_ptr<ir::tensor::TensorValue>> outputs_ir = ir::tensor::wrapTensors2TensorIR(ir_ctx.get(), outputs);

  // link outputs to subgraph.
  for (const auto& output_ir : outputs_ir) {
    (*this_graph_ir)-- > output_ir;
    (*call_op)-- > output_ir;
    this_graph_ir->getTopRegion()->outputs().push_back(output_ir);
  }

  // create return op
  {
    auto guard = ir::IRWriterGuard(ir_ctx, this_graph_ir->getTopRegion());
    std::vector<ir::val_ptr_t> vals;
    vals.reserve(outputs_ir.size());
    for (auto& o : outputs_ir) vals.push_back(o);
    ir_ctx->create<ir::cf::ReturnOp>(vals);
  }

  // return the outputs.
  return outputs;
}

ParameterFile::ptr_t Module::params(ModelFileVersion v) { return impl_->params(v); }

}  // namespace mllm::nn
