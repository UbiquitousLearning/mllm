// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <memory>
#include <nlohmann/json.hpp>

#include "mllm/mllm.hpp"

struct QuantizeDescriptor {
  std::string pattern;
  nlohmann::json hints;
};

class QuantizeImpl {
 public:
  using ptr_t = std::shared_ptr<QuantizeImpl>;

  virtual bool match(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) = 0;

  virtual mllm::ParameterFile::ptr_t perform(const QuantizeDescriptor& desc, mllm::ParameterFile::ptr_t params) = 0;
};
