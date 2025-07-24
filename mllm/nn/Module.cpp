/**
 * @file Module.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-24
 *
 */
#include "mllm/nn/Module.hpp"
#include "mllm/nn/AbstractNnNode.hpp"

namespace mllm::nn {
ModuleImpl::ModuleImpl() : AbstractNnNode(AbstractNnNodeTypes::kModule) {}

void ModuleImpl::load(const ParameterFile::ptr_t& param_file) {
  param_file_ = param_file;
  auto& h = refChildNodes();
  for (auto& hb : h) {
    switch (hb->getType()) {
      case AbstractNnNodeTypes::kModule: std::static_pointer_cast<ModuleImpl>(hb)->load(param_file); break;
      case AbstractNnNodeTypes::kLayer: std::static_pointer_cast<LayerImpl>(hb)->load(param_file); break;
    }
  }
}

ParameterFile::ptr_t ModuleImpl::params() const { return param_file_; }

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

void Module::reg_(const std::string& name, const AbstractNnNode::ptr_t& node) {
  // TODO
}

Module& Module::load(const ParameterFile::ptr_t& param_file) {
  // TODO
  return *this;
}

}  // namespace mllm::nn
