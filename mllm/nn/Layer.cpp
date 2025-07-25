/**
 * @file Layer.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-24
 *
 */
#include "mllm/nn/Layer.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/engine/Context.hpp"

namespace mllm::nn {

ParameterFile::ptr_t LayerImpl::refParams() { return parameter_loader_; }

void LayerImpl::load(const ParameterFile::ptr_t& ploader) {
  parameter_loader_ = ploader;
  Context::instance().thisThread()->layer_ops_table[getAbsoluteName()]->load(ploader);
}

void LayerImpl::to(DeviceTypes device_type) {
  auto& ctx = Context::instance();
  ctx.thisThread()->layer_ops_table.remove(getAbsoluteName());
  ctx.thisThread()->layer_ops_table.reg(getAbsoluteName(), ctx.getBackend(device_type)->createOp(op_type_, options_));
  ctx.thisThread()->layer_ops_table[getAbsoluteName()]->setName(getAbsoluteName());
  if (parameter_loader_) {
    // reload param to current op
    ctx.thisThread()->layer_ops_table[getAbsoluteName()]->load(parameter_loader_);
  }
  device_type_ = device_type;
}

OpTypes LayerImpl::opType() const { return op_type_; }

BaseOpOptionsBase& LayerImpl::refOptions() { return options_; }

void LayerImpl::__fmt_print(std::stringstream& ss) {
  for (int i = 0; i < getDepth() * 4; i++) { ss << " "; }
  ss << getAbsoluteName() << ", device: " << deviceTypes2Str(device_type_);
}

LayerImpl::ptr_t Layer::impl() const { return impl_; }

OpTypes Layer::opType() const { return impl()->opType(); }

BaseOpOptionsBase& Layer::refOptions() { return impl()->refOptions(); }

Layer& Layer::to(DeviceTypes device_type) {
  impl()->to(device_type);
  return *this;
}

void Layer::__fmt_print(std::stringstream& ss) { __fmt_print(ss); }

}  // namespace mllm::nn
