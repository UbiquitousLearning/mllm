// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/Layer.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/ParameterFile.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::nn {
void LayerImpl::load(const ParameterFile::ptr_t& ploader) { instanced_op_->load(ploader); }

void LayerImpl::to(DeviceTypes device_type) {
  auto& ctx = Context::instance();
  instanced_op_ = nullptr;
  instanced_op_ = ctx.getBackend(device_type)->createOp(op_type_, options_);
  instanced_op_->setName(getAbsoluteName());

  auto temp_params_loader = ParameterFile::create();

  auto params = instanced_op_->getParams();
  for (auto& param : *params) {
    auto t = param.second;
    if (t) { temp_params_loader->push(param.first, t.to(device_type)); }
  }
  instanced_op_->load(temp_params_loader);
  device_type_ = device_type;
}

OpTypes LayerImpl::opType() const { return op_type_; }

BaseOpOptionsBase& LayerImpl::refOptions() { return options_; }

void LayerImpl::__fmt_print(std::stringstream& ss) {
  for (int i = 0; i < getDepth() * 4; i++) { ss << " "; }
  ss << getAbsoluteName() << ", device: " << deviceTypes2Str(device_type_);
}

BaseOp::ptr_t LayerImpl::getInstancedOp() { return instanced_op_; }

void LayerImpl::setInstancedOp(const BaseOp::ptr_t& op) { instanced_op_ = op; }

LayerImpl::ptr_t Layer::impl() const { return impl_; }

OpTypes Layer::opType() const { return impl()->opType(); }

BaseOpOptionsBase& Layer::refOptions() { return impl()->refOptions(); }

Layer& Layer::to(DeviceTypes device_type) {
  impl()->to(device_type);
  return *this;
}

void Layer::__fmt_print(std::stringstream& ss) { __fmt_print(ss); }

}  // namespace mllm::nn
