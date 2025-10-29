// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class Conv2DOpImplType {
  kDefault = 0,
};

struct Conv2DOpOptions : public BaseOpOptions<Conv2DOpOptions> {
  int32_t in_channels;
  int32_t out_channels;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  std::vector<int32_t> padding;
  std::vector<int32_t> dilation;
  bool bias = true;
  Conv2DOpImplType impl_type = Conv2DOpImplType::kDefault;
};

inline Conv2DOpImplType str2Conv2DOpImplType(const std::string& str) {
  static const std::unordered_map<std::string, Conv2DOpImplType> map = {{"Default", Conv2DOpImplType::kDefault}};

  auto it = map.find(str);
  if (it != map.end()) { return it->second; }

  // Return default if not found
  return Conv2DOpImplType::kDefault;
}

inline std::string Conv2DOpImplType2Str(Conv2DOpImplType type) {
  static const std::unordered_map<Conv2DOpImplType, std::string> map = {{Conv2DOpImplType::kDefault, "Default"}};

  auto it = map.find(type);
  if (it != map.end()) return it->second;
  return "Default";
}

class Conv2DOp : public BaseOp {
 public:
  explicit Conv2DOp(const Conv2DOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline Conv2DOpOptions& options() { return options_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  Conv2DOpOptions options_;
};

}  // namespace mllm::aops
