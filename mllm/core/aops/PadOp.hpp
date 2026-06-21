// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

enum class PadMode : uint8_t {
  kConstant = 0,
  kReflect = 1,
  kReplicate = 2,
  kCircular = 3,
};

struct PadOpOptions : public BaseOpOptions<PadOpOptions> {
  std::vector<int32_t> pad;          // padding sizes, starting from the last dimension
  PadMode mode{PadMode::kConstant};  // padding mode
  float value{0.0f};                 // padding value for constant mode
};

class PadOp : public BaseOp {
 public:
  explicit PadOp(const PadOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const PadOpOptions& options() const { return options_; }

 protected:
  PadOpOptions options_;
};

}  // namespace mllm::aops
