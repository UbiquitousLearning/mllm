// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <regex>

#include "driver.hpp"

QuantizeDriver::QuantizeDriver(const std::vector<mllm::ParameterFile::ptr_t>& params, mllm::ConfigFile& cfg)
    : params_(params), cfg_(cfg) {}

void QuantizeDriver::registerQuantizeImpl(const QuantizeImpl::ptr_t& impl) { quantize_impls_.emplace_back(impl); }

void QuantizeDriver::operator()() {
  for (auto& item : cfg_.data().items()) {
    const std::string& pattern = item.key();
    nlohmann::json hints = item.value()["hints"];

    QuantizeDescriptor desc;
    desc.pattern = pattern;
    desc.hints = hints;

    std::regex re(pattern);

    for (auto& param : params_) {
      for (auto [name, tensor] : *param) {
        if (std::regex_match(name, re)) {
          for (auto& impl : quantize_impls_) {
            if (impl->match(desc, tensor)) {
              tensor = impl->perform(desc, tensor);
              param->remove(name);
              param->push(name, tensor);
              break;
            }
          }
        }
      }
    }
  }
}
