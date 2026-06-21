// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <vector>

#include "schema/quantize_base.hpp"

class QuantizeDriver {
 public:
  QuantizeDriver(const mllm::ParameterFile::ptr_t& params, mllm::ConfigFile& cfg);

  void registerQuantizeImpl(const QuantizeImpl::ptr_t& impl);

  void operator()();

 private:
  mllm::ConfigFile& cfg_;
  mllm::ParameterFile::ptr_t params_;

  std::vector<QuantizeImpl::ptr_t> quantize_impls_;
};
