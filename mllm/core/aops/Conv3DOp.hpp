// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class Conv3DOpImplType {
  kDefault = 0,
};

struct Conv3DOpOptions : public BaseOpOptions<Conv3DOpOptions> {
  int32_t in_channels;
  int32_t out_channels;
  std::vector<int32_t> kernel_size;
  std::vector<int32_t> stride;
  bool bias = true;
  Conv3DOpImplType impl_type = Conv3DOpImplType::kDefault;
};

inline Conv3DOpImplType str2Conv3DOpImplType(const std::string& str) {
  static const std::unordered_map<std::string, Conv3DOpImplType> map = {{"Default", Conv3DOpImplType::kDefault}};

  auto it = map.find(str);
  if (it != map.end()) { return it->second; }

  // Return default if not found
  return Conv3DOpImplType::kDefault;
}

inline std::string conv3DOpImplType2Str(Conv3DOpImplType type) {
  static const std::unordered_map<Conv3DOpImplType, std::string> map = {{Conv3DOpImplType::kDefault, "Default"}};

  auto it = map.find(type);
  if (it != map.end()) return it->second;
  return "Default";
}

class Conv3DOp : public BaseOp {
 public:
  explicit Conv3DOp(const Conv3DOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  ParameterFile::ptr_t getParams() override;

  inline Tensor& weight() { return weight_; }

  inline Tensor& bias() { return bias_; }

  inline const Conv3DOpOptions& options() const { return options_; }

 protected:
  Tensor weight_;
  Tensor bias_;
  Conv3DOpOptions options_;
};

}  // namespace mllm::aops
