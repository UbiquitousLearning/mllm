// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <regex>
#include <unordered_map>

#include "mllm/core/ParameterFile.hpp"
#include "mllm/utils/ProgressBar.hpp"

#include "driver.hpp"

namespace {
std::string extractGroupName(const std::string& tensor_name) {
  size_t last_dot = tensor_name.rfind('.');
  if (last_dot != std::string::npos) { return tensor_name.substr(0, last_dot); }
  return tensor_name;
}
}  // namespace

QuantizeDriver::QuantizeDriver(const mllm::ParameterFile::ptr_t& params, mllm::ConfigFile& cfg) : params_(params), cfg_(cfg) {}

void QuantizeDriver::registerQuantizeImpl(const QuantizeImpl::ptr_t& impl) { quantize_impls_.emplace_back(impl); }

void QuantizeDriver::operator()() {
  std::vector<std::pair<QuantizeDescriptor, std::unordered_map<std::string, mllm::ParameterFile::ptr_t>>> param_groups;

  for (auto& item : cfg_.data().items()) {
    std::unordered_map<std::string, mllm::ParameterFile::ptr_t> sub_group;
    const std::string& pattern = item.key();
    nlohmann::json hints = item.value()["hints"];
    QuantizeDescriptor desc;
    desc.pattern = pattern;
    desc.hints = hints;
    std::regex re(pattern);

    for (const auto& [name, tensor] : *params_) {
      if (std::regex_match(name, re)) {
        std::string layer_name = extractGroupName(name);
        if (sub_group.find(layer_name) == sub_group.end()) {
          sub_group[layer_name] = mllm::ParameterFile::create(params_->version());
        }
        sub_group[layer_name]->push(name, tensor);
      }
    }

    param_groups.emplace_back(desc, sub_group);
  }

  // Calculate total numbers of parameters
  int cnt = 0;
  for (auto& [_, p] : param_groups) { cnt += p.size(); }

  mllm::ProgressBar progress_bar("Quantizing parameters", cnt);
  cnt = 0;
  for (auto& [desc, p] : param_groups) {
    for (auto& [name, tensors] : p) {
      for (auto& impl : quantize_impls_) {
        if (impl->match(desc, tensors)) {
          auto new_tensors = impl->perform(desc, tensors);
          for (auto& nt : *new_tensors) {
            // E.g.: For Embedding. We need fp32 embedding and kai's quant embedding both.
            // Do not replace embedding weights, but rename it and add a new one.
            if (!desc.hints["replace"].is_null() && !desc.hints["replace"]) {
              params_->push(desc.hints["rename"], nt.second);
            } else {
              auto name = nt.first;
              if (!desc.hints["rename"].is_null()) { name = desc.hints["rename"]; }
              params_->remove(nt.first);
              params_->push(name, nt.second);
            }
          }
          // E.g.: kai will pack bias and weight into weight. We need to remove bias.
          // bias is in tensors, but not in new_tensors, we need to remove it.
          for (auto& nt : *tensors) {
            if (!new_tensors->has(nt.first)) { params_->remove(nt.first); }
          }
          break;
        }
      }
      progress_bar.update(++cnt);
    }
  }
}
