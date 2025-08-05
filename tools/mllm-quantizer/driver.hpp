// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <vector>
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

  virtual bool match(const QuantizeDescriptor& desc, mllm::Tensor& t) = 0;

  virtual mllm::Tensor perform(const QuantizeDescriptor& desc, mllm::Tensor& t) = 0;
};

class QuantizeDriver {
 public:
  QuantizeDriver(const std::vector<mllm::ParameterFile::ptr_t>& params, mllm::ConfigFile& cfg);

  void registerQuantizeImpl(const QuantizeImpl::ptr_t& impl);

  void operator()();

 private:
  mllm::ConfigFile& cfg_;
  std::vector<mllm::ParameterFile::ptr_t> params_;

  std::vector<QuantizeImpl::ptr_t> quantize_impls_;
};
